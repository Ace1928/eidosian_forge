from eventlet import event as _event
class metaphore:
    """This is sort of an inverse semaphore: a counter that starts at 0 and
    waits only if nonzero. It's used to implement a "wait for all" scenario.

    >>> from eventlet import coros, spawn_n
    >>> count = coros.metaphore()
    >>> count.wait()
    >>> def decrementer(count, id):
    ...     print("{0} decrementing".format(id))
    ...     count.dec()
    ...
    >>> _ = spawn_n(decrementer, count, 'A')
    >>> _ = spawn_n(decrementer, count, 'B')
    >>> count.inc(2)
    >>> count.wait()
    A decrementing
    B decrementing
    """

    def __init__(self):
        self.counter = 0
        self.event = _event.Event()
        self.event.send()

    def inc(self, by=1):
        """Increment our counter. If this transitions the counter from zero to
        nonzero, make any subsequent :meth:`wait` call wait.
        """
        assert by > 0
        self.counter += by
        if self.counter == by:
            self.event.reset()

    def dec(self, by=1):
        """Decrement our counter. If this transitions the counter from nonzero
        to zero, a current or subsequent wait() call need no longer wait.
        """
        assert by > 0
        self.counter -= by
        if self.counter <= 0:
            self.counter = 0
            self.event.send()

    def wait(self):
        """Suspend the caller only if our count is nonzero. In that case,
        resume the caller once the count decrements to zero again.
        """
        self.event.wait()