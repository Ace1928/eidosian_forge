import re
from . import urlutils
from .hooks import Hooks
Determine a fully qualified URL from a location string.

    This will try to interpret location as both a URL and a directory path. It
    will also lookup the location in directories.

    :param location: Unicode or byte string object with a location
    :param purpose: Intended method of access (None, 'read' or 'write')
    :raise InvalidURL: If the location is already a URL, but not valid.
    :return: Byte string with resulting URL
    