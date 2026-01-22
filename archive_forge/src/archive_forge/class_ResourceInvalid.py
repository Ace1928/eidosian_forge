from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class ResourceInvalid(ResourceError):
    """Resource has the wrong type."""
    default_message = "resource '{path}' is invalid for this operation"