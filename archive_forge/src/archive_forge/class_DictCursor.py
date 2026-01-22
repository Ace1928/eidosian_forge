import re
import warnings
from . import err
class DictCursor(DictCursorMixin, Cursor):
    """A cursor which returns results as a dictionary"""