from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class TextTypes(_TextTypes):
    """Defines text types that can be used for styling text."""
    RESOURCE_NAME = 1
    URL = 2
    USER_INPUT = 3
    COMMAND = 4
    INFO = 5
    URI = 6
    OUTPUT = 7
    PT_SUCCESS = 8
    PT_FAILURE = 9