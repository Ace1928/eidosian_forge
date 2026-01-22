from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
class _InitialLengthAdapter(Adapter):
    """ A standard Construct adapter that expects a sub-construct
        as a struct with one or two values (first, second).
    """

    def _decode(self, obj, context):
        if obj.first < 4294967040:
            context['is64'] = False
            return obj.first
        elif obj.first == 4294967295:
            context['is64'] = True
            return obj.second
        else:
            raise ConstructError('Failed decoding initial length for %X' % obj.first)