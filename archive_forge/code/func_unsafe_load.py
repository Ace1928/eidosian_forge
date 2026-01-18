from .error import *
from .tokens import *
from .events import *
from .nodes import *
from .loader import *
from .dumper import *
import io
def unsafe_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.

    Resolve all tags, even those known to be
    unsafe on untrusted input.
    """
    return load(stream, UnsafeLoader)