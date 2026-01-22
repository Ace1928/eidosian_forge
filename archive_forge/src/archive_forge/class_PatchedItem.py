import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string
class PatchedItem(ContextManager):
    """Context manager that temporary replaces an object item using :meth:`~object.__setitem__()`."""

    def __init__(self, obj, item, value):
        """
        Initialize a :class:`PatchedItem` object.

        :param obj: The object to patch.
        :param item: The item to patch.
        :param value: The value to set.
        """
        self.object_to_patch = obj
        self.item_to_patch = item
        self.patched_value = value
        self.original_value = NOTHING

    def __enter__(self):
        """
        Replace (patch) the item.

        :returns: The object whose item was patched.
        """
        super(PatchedItem, self).__enter__()
        try:
            self.original_value = self.object_to_patch[self.item_to_patch]
        except KeyError:
            self.original_value = NOTHING
        self.object_to_patch[self.item_to_patch] = self.patched_value
        return self.object_to_patch

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Restore the item to its original value."""
        super(PatchedItem, self).__exit__(exc_type, exc_value, traceback)
        if self.original_value is NOTHING:
            del self.object_to_patch[self.item_to_patch]
        else:
            self.object_to_patch[self.item_to_patch] = self.original_value