import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def _raise_context_error(self, message):
    """Try to check for OpenAL error and raise that, and then
        definitely raise an error with the given message.
        """
    self.check_context_error(message)
    raise OpenALException(message)