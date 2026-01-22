import itertools
import sys
from fixtures.callmany import (
class MethodFixture(Fixture):
    """An adapter to use a function as a Fixture.

    Typically used when an existing object exists but you wish to use it as a
    Fixture (e.g. because fixtures are in use in your test suite and this will
    fit in better).

    To adapt an object with setUp / tearDown methods:
    fixture = MethodFixture(object)
    If setUp / tearDown / reset are missing, they simply won't be called.

    The object is exposed on fixture.obj.

    To adapt an object with differently named setUp and cleanUp methods:
    fixture = MethodFixture(object, setup=object.mySetUp,
        teardown=object.myTearDown)

    With a differently named reset function:
    fixture = MethodFixture(object, reset=object.myReset)

    :ivar obj: The object which is being wrapped.
    """

    def __init__(self, obj, setup=None, cleanup=None, reset=None):
        """Create a MethodFixture.

        :param obj: The object to wrap. Exposed as fixture.obj
        :param setup: A method which takes no parameters. e.g.
            def setUp(self):
                self.value = 42
            If setup is not supplied, and the object has a setUp method, that
            method is used, otherwise nothing will happen during fixture.setUp.
        :param cleanup: Optional method to cleanup the object's state. If
            not supplied the method 'tearDown' is used if it exists.
        :param reset: Optional method to reset the wrapped object for use.
            If not supplied, then the method 'reset' is used if it exists,
            otherwise cleanUp and setUp are called as per Fixture.reset().
        """
        super(MethodFixture, self).__init__()
        self.obj = obj
        if setup is None:
            setup = getattr(obj, 'setUp', None)
            if setup is None:
                setup = lambda: None
        self._setup = setup
        if cleanup is None:
            cleanup = getattr(obj, 'tearDown', None)
            if cleanup is None:
                cleanup = lambda: None
        self._cleanup = cleanup
        if reset is None:
            reset = getattr(obj, 'reset', None)
        self._reset = reset

    def _setUp(self):
        self._setup()

    def cleanUp(self):
        super(MethodFixture, self).cleanUp()
        self._cleanup()

    def reset(self):
        if self._reset is None:
            super(MethodFixture, self).reset()
        else:
            self._reset()