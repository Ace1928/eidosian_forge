from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
@property
def annotation_string(self):
    return self._return_string