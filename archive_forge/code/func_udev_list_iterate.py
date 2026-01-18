import errno
import os
import stat
import sys
from subprocess import check_output
def udev_list_iterate(libudev, entry):
    """
    Iteration helper for udev list entry objects.

    Yield a tuple ``(name, value)``.  ``name`` and ``value`` are bytestrings
    containing the name and the value of the list entry.  The exact contents
    depend on the list iterated over.
    """
    while entry:
        name = libudev.udev_list_entry_get_name(entry)
        value = libudev.udev_list_entry_get_value(entry)
        yield (name, value)
        entry = libudev.udev_list_entry_get_next(entry)