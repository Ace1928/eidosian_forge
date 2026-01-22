import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
class _AssertTraitChangesContext(object):
    """ A context manager used to implement the trait change assert methods.

    Notes
    -----
    Checking if the provided xname corresponds to valid traits in the class
    is not implemented yet.

    Parameters
    ----------
    obj : HasTraits
        The HasTraits class instance whose class trait will change.

    xname : str
        The extended trait name of trait changes to listen to.

    count : int, optional
        The expected number of times the event should be fired. When None
        (default value) there is no check for the number of times the
        change event was fired.

    test_case : TestCase
        A unittest TestCase where to raise the failureException if
        necessary.

    Attributes
    ----------
    obj : HasTraits
        The HasTraits class instance whose class trait will change.

    xname : str
        The extended trait name of trait changes to listen to.

    count : int, optional
        The expected number of times the event should be fired. When None
        (default value) there is no check for the number of times the
        change event was fired.

    events : list of tuples
        A list with tuple elements containing the arguments of an
        `on_trait_change` event signature (<object>, <name>, <old>, <new>).

    Raises
    ------
    AssertionError :
          When the desired number of trait changed did not take place or when
          `count = None` and no trait change took place.

    """

    def __init__(self, obj, xname, count, test_case):
        self.obj = obj
        self.xname = xname
        self.count = count
        self.event = None
        self.events = []
        self.failureException = test_case.failureException

    def _listener(self, obj, name, old, new):
        """ Dummy trait listener.
        """
        self.event = (obj, name, old, new)
        self.events.append(self.event)

    def __enter__(self):
        """ Bind the trait listener.
        """
        self.obj.on_trait_change(self._listener, self.xname)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """ Remove the trait listener.
        """
        if exc_type is not None:
            return False
        self.obj.on_trait_change(self._listener, self.xname, remove=True)
        if self.count is not None and len(self.events) != self.count:
            msg = 'Change event for {0} was fired {1} times instead of {2}'
            items = (self.xname, len(self.events), self.count)
            raise self.failureException(msg.format(*items))
        elif self.count is None and (not self.events):
            msg = 'A change event was not fired for: {0}'.format(self.xname)
            raise self.failureException(msg)
        return False