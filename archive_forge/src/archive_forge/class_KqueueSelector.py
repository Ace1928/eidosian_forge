from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
class KqueueSelector(_BaseSelectorImpl):
    """Kqueue-based selector."""

    def __init__(self):
        super().__init__()
        self._selector = select.kqueue()
        self._max_events = 0

    def fileno(self):
        return self._selector.fileno()

    def register(self, fileobj, events, data=None):
        key = super().register(fileobj, events, data)
        try:
            if events & EVENT_READ:
                kev = select.kevent(key.fd, select.KQ_FILTER_READ, select.KQ_EV_ADD)
                self._selector.control([kev], 0, 0)
                self._max_events += 1
            if events & EVENT_WRITE:
                kev = select.kevent(key.fd, select.KQ_FILTER_WRITE, select.KQ_EV_ADD)
                self._selector.control([kev], 0, 0)
                self._max_events += 1
        except:
            super().unregister(fileobj)
            raise
        return key

    def unregister(self, fileobj):
        key = super().unregister(fileobj)
        if key.events & EVENT_READ:
            kev = select.kevent(key.fd, select.KQ_FILTER_READ, select.KQ_EV_DELETE)
            self._max_events -= 1
            try:
                self._selector.control([kev], 0, 0)
            except OSError:
                pass
        if key.events & EVENT_WRITE:
            kev = select.kevent(key.fd, select.KQ_FILTER_WRITE, select.KQ_EV_DELETE)
            self._max_events -= 1
            try:
                self._selector.control([kev], 0, 0)
            except OSError:
                pass
        return key

    def select(self, timeout=None):
        timeout = None if timeout is None else max(timeout, 0)
        max_ev = self._max_events or 1
        ready = []
        try:
            kev_list = self._selector.control(None, max_ev, timeout)
        except InterruptedError:
            return ready
        for kev in kev_list:
            fd = kev.ident
            flag = kev.filter
            events = 0
            if flag == select.KQ_FILTER_READ:
                events |= EVENT_READ
            if flag == select.KQ_FILTER_WRITE:
                events |= EVENT_WRITE
            key = self._key_from_fd(fd)
            if key:
                ready.append((key, events & key.events))
        return ready

    def close(self):
        self._selector.close()
        super().close()