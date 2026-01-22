import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
class TokenMergeFilter(Filter):
    """Merges consecutive tokens with the same token type in the output
    stream of a lexer.

    .. versionadded:: 1.2
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)

    def filter(self, lexer, stream):
        current_type = None
        current_value = None
        for ttype, value in stream:
            if ttype is current_type:
                current_value += value
            else:
                if current_type is not None:
                    yield (current_type, current_value)
                current_type = ttype
                current_value = value
        if current_type is not None:
            yield (current_type, current_value)