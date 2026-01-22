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
class PatchedAttribute(ContextManager):
    """Context manager that temporary replaces an object attribute using :func:`setattr()`."""

    def __init__(self, obj, name, value):
        """
        Initialize a :class:`PatchedAttribute` object.

        :param obj: The object to patch.
        :param name: An attribute name.
        :param value: The value to set.
        """
        self.object_to_patch = obj
        self.attribute_to_patch = name
        self.patched_value = value
        self.original_value = NOTHING

    def __enter__(self):
        """
        Replace (patch) the attribute.

        :returns: The object whose attribute was patched.
        """
        super(PatchedAttribute, self).__enter__()
        self.original_value = getattr(self.object_to_patch, self.attribute_to_patch, NOTHING)
        setattr(self.object_to_patch, self.attribute_to_patch, self.patched_value)
        return self.object_to_patch

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Restore the attribute to its original value."""
        super(PatchedAttribute, self).__exit__(exc_type, exc_value, traceback)
        if self.original_value is NOTHING:
            delattr(self.object_to_patch, self.attribute_to_patch)
        else:
            setattr(self.object_to_patch, self.attribute_to_patch, self.original_value)