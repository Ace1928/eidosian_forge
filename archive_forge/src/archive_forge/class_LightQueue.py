import sys
import heapq
import collections
import traceback
from eventlet.event import Event
from eventlet.greenthread import getcurrent
from eventlet.hubs import get_hub
import queue as Stdlib_Queue
from eventlet.timeout import Timeout
class LightQueue:
    """
    This is a variant of Queue that behaves mostly like the standard
    :class:`Stdlib_Queue`.  It differs by not supporting the
    :meth:`task_done <Stdlib_Queue.task_done>` or
    :meth:`join <Stdlib_Queue.join>` methods, and is a little faster for
    not having that overhead.
    """

    def __init__(self, maxsize=None):
        if maxsize is None or maxsize < 0:
            self.maxsize = None
        else:
            self.maxsize = maxsize
        self.getters = set()
        self.putters = set()
        self._event_unlock = None
        self._init(maxsize)

    def _init(self, maxsize):
        self.queue = collections.deque()

    def _get(self):
        return self.queue.popleft()

    def _put(self, item):
        self.queue.append(item)

    def __repr__(self):
        return '<%s at %s %s>' % (type(self).__name__, hex(id(self)), self._format())

    def __str__(self):
        return '<%s %s>' % (type(self).__name__, self._format())

    def _format(self):
        result = 'maxsize=%r' % (self.maxsize,)
        if getattr(self, 'queue', None):
            result += ' queue=%r' % self.queue
        if self.getters:
            result += ' getters[%s]' % len(self.getters)
        if self.putters:
            result += ' putters[%s]' % len(self.putters)
        if self._event_unlock is not None:
            result += ' unlocking'
        return result

    def qsize(self):
        """Return the size of the queue."""
        return len(self.queue)

    def resize(self, size):
        """Resizes the queue's maximum size.

        If the size is increased, and there are putters waiting, they may be woken up."""
        if self.maxsize is not None and (size is None or size > self.maxsize):
            self._schedule_unlock()
        self.maxsize = size

    def putting(self):
        """Returns the number of greenthreads that are blocked waiting to put
        items into the queue."""
        return len(self.putters)

    def getting(self):
        """Returns the number of greenthreads that are blocked waiting on an
        empty queue."""
        return len(self.getters)

    def empty(self):
        """Return ``True`` if the queue is empty, ``False`` otherwise."""
        return not self.qsize()

    def full(self):
        """Return ``True`` if the queue is full, ``False`` otherwise.

        ``Queue(None)`` is never full.
        """
        return self.maxsize is not None and self.qsize() >= self.maxsize

    def put(self, item, block=True, timeout=None):
        """Put an item into the queue.

        If optional arg *block* is true and *timeout* is ``None`` (the default),
        block if necessary until a free slot is available. If *timeout* is
        a positive number, it blocks at most *timeout* seconds and raises
        the :class:`Full` exception if no free slot was available within that time.
        Otherwise (*block* is false), put an item on the queue if a free slot
        is immediately available, else raise the :class:`Full` exception (*timeout*
        is ignored in that case).
        """
        if self.maxsize is None or self.qsize() < self.maxsize:
            self._put(item)
            if self.getters:
                self._schedule_unlock()
        elif not block and get_hub().greenlet is getcurrent():
            while self.getters:
                getter = self.getters.pop()
                if getter:
                    self._put(item)
                    item = self._get()
                    getter.switch(item)
                    return
            raise Full
        elif block:
            waiter = ItemWaiter(item, block)
            self.putters.add(waiter)
            timeout = Timeout(timeout, Full)
            try:
                if self.getters:
                    self._schedule_unlock()
                result = waiter.wait()
                assert result is waiter, 'Invalid switch into Queue.put: %r' % (result,)
                if waiter.item is not _NONE:
                    self._put(item)
            finally:
                timeout.cancel()
                self.putters.discard(waiter)
        elif self.getters:
            waiter = ItemWaiter(item, block)
            self.putters.add(waiter)
            self._schedule_unlock()
            result = waiter.wait()
            assert result is waiter, 'Invalid switch into Queue.put: %r' % (result,)
            if waiter.item is not _NONE:
                raise Full
        else:
            raise Full

    def put_nowait(self, item):
        """Put an item into the queue without blocking.

        Only enqueue the item if a free slot is immediately available.
        Otherwise raise the :class:`Full` exception.
        """
        self.put(item, False)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args *block* is true and *timeout* is ``None`` (the default),
        block if necessary until an item is available. If *timeout* is a positive number,
        it blocks at most *timeout* seconds and raises the :class:`Empty` exception
        if no item was available within that time. Otherwise (*block* is false), return
        an item if one is immediately available, else raise the :class:`Empty` exception
        (*timeout* is ignored in that case).
        """
        if self.qsize():
            if self.putters:
                self._schedule_unlock()
            return self._get()
        elif not block and get_hub().greenlet is getcurrent():
            while self.putters:
                putter = self.putters.pop()
                if putter:
                    putter.switch(putter)
                    if self.qsize():
                        return self._get()
            raise Empty
        elif block:
            waiter = Waiter()
            timeout = Timeout(timeout, Empty)
            try:
                self.getters.add(waiter)
                if self.putters:
                    self._schedule_unlock()
                try:
                    return waiter.wait()
                except:
                    self._schedule_unlock()
                    raise
            finally:
                self.getters.discard(waiter)
                timeout.cancel()
        else:
            raise Empty

    def get_nowait(self):
        """Remove and return an item from the queue without blocking.

        Only get an item if one is immediately available. Otherwise
        raise the :class:`Empty` exception.
        """
        return self.get(False)

    def _unlock(self):
        try:
            while True:
                if self.qsize() and self.getters:
                    getter = self.getters.pop()
                    if getter:
                        try:
                            item = self._get()
                        except:
                            getter.throw(*sys.exc_info())
                        else:
                            getter.switch(item)
                elif self.putters and self.getters:
                    putter = self.putters.pop()
                    if putter:
                        getter = self.getters.pop()
                        if getter:
                            item = putter.item
                            putter.item = _NONE
                            self._put(item)
                            item = self._get()
                            getter.switch(item)
                            putter.switch(putter)
                        else:
                            self.putters.add(putter)
                elif self.putters and (self.getters or self.maxsize is None or self.qsize() < self.maxsize):
                    putter = self.putters.pop()
                    putter.switch(putter)
                elif self.putters and (not self.getters):
                    full = [p for p in self.putters if not p.block]
                    if not full:
                        break
                    for putter in full:
                        self.putters.discard(putter)
                        get_hub().schedule_call_global(0, putter.greenlet.throw, Full)
                else:
                    break
        finally:
            self._event_unlock = None

    def _schedule_unlock(self):
        if self._event_unlock is None:
            self._event_unlock = get_hub().schedule_call_global(0, self._unlock)