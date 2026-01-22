from __future__ import print_function, absolute_import, division
import unittest
class Dealloc:
    """
            An object with a ``__del__`` method. When it starts getting deallocated
            from a deferred trash can run, it switches greenlets, allocates more objects
            which then also go in the trash can. If we don't save state appropriately,
            nesting gets out of order and we can crash the interpreter.
            """
    SPAWNED = False
    BG_RAN = False
    BG_GLET = None
    CREATED = 0
    DESTROYED = 0
    DESTROYED_BG = 0

    def __init__(self, sequence_number):
        """
                :param sequence_number: The ordinal of this object during
                   one particular creation run. This is used to detect (guess, really)
                   when we have entered the trash can's deferred deallocation.
                """
        self.i = sequence_number
        Dealloc.CREATED += 1

    def __del__(self):
        if self.i == TRASH_UNWIND_LEVEL and (not self.SPAWNED):
            Dealloc.SPAWNED = greenlet.getcurrent()
            other = Dealloc.BG_GLET = greenlet.greenlet(background_greenlet)
            x = other.switch()
            assert x == 42
            del other
        elif self.i == 40 and greenlet.getcurrent() is not main:
            Dealloc.BG_RAN = True
            try:
                main.switch(42)
            except greenlet.GreenletExit as ex:
                Dealloc.BG_RAN = type(ex)
                del ex
        Dealloc.DESTROYED += 1
        if greenlet.getcurrent() is not main:
            Dealloc.DESTROYED_BG += 1