import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
@contextlib.contextmanager
def assertTraitChangesAsync(self, obj, trait, count=1, timeout=5.0):
    """ Assert an object trait eventually changes.

        Context manager used to assert that the given trait changes at
        least `count` times within the given timeout, as a result of
        execution of the body of the corresponding with block.

        The trait changes are permitted to occur asynchronously.

        **Example usage**::

            with self.assertTraitChangesAsync(my_object, 'SomeEvent', count=4):
                <do stuff that should cause my_object.SomeEvent to be
                fired at least 4 times within the next 5 seconds>


        Parameters
        ----------
        obj : HasTraits
            The HasTraits class instance whose class trait will change.

        trait : str
            The extended trait name of trait changes to listen to.

        count : int, optional
            The expected number of times the event should be fired.

        timeout : float or None, optional
            The amount of time in seconds to wait for the specified number
            of changes. None can be used to indicate no timeout.

        """
    collector = _TraitsChangeCollector(obj=obj, trait_name=trait)
    collector.start_collecting()
    try:
        yield collector
        try:
            wait_for_condition(condition=lambda obj: obj.event_count >= count, obj=collector, trait='event_count_updated', timeout=timeout)
        except RuntimeError:
            actual_event_count = collector.event_count
            msg = 'Expected {0} event on {1} to be fired at least {2} times, but the event was only fired {3} times before timeout ({4} seconds).'.format(trait, obj, count, actual_event_count, timeout)
            self.fail(msg)
    finally:
        collector.stop_collecting()