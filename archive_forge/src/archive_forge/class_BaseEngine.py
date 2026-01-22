import sys
import datetime
from collections import namedtuple
class BaseEngine(object):
    """Base class for Pyzor engines."""
    absolute_source = True
    handles_one_step = False

    def __iter__(self):
        """Iterate over all keys"""
        raise NotImplementedError()

    def iteritems(self):
        """Iterate over pairs of (key, record)."""
        raise NotImplementedError()

    def items(self):
        """Return a list of (key, record)."""
        raise NotImplementedError()

    def __getitem__(self, key):
        """Get the record for this corresponding key."""
        raise NotImplementedError()

    def __setitem__(self, key, value):
        """Set the record for this corresponding key. 'value' should be a
        instance of the ``Record`` class.
        """
        raise NotImplementedError()

    def __delitem__(self, key):
        """Remove the corresponding record from the database."""
        raise NotImplementedError()

    def report(self, keys):
        """Report the corresponding key as spam, incrementing the report count.

        Engines that implement don't implement this method should have
        handles_one_step set to False.
        """
        raise NotImplementedError()

    def whitelist(self, keys):
        """Report the corresponding key as ham, incrementing the whitelist
        count.

        Engines that implement don't implement this method should have
        handles_one_step set to False.
        """
        raise NotImplementedError()

    @classmethod
    def get_prefork_connections(cls, fn, mode, max_age=None):
        """Yields an unlimited number of partial functions that return a new
        engine instance, suitable for using toghether with the Pre-Fork server.
        """
        raise NotImplementedError()