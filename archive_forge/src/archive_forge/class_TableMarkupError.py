import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
class TableMarkupError(DataError):
    """
    Raise if there is any problem with table markup.

    The keyword argument `offset` denotes the offset of the problem
    from the table's start line.
    """

    def __init__(self, *args, **kwargs):
        self.offset = kwargs.pop('offset', 0)
        DataError.__init__(self, *args)