from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
class FormattedEntry(Construct):

    def __init__(self, name, structs, format_field):
        Construct.__init__(self, name)
        self.structs = structs
        self.format_field = format_field

    def _parse(self, stream, context):
        if self.format_field + '_parser' in context:
            parser = context[self.format_field + '_parser']
        else:
            fields = tuple((Rename(f.content_type, self.structs.Dwarf_dw_form[f.form]) for f in context[self.format_field]))
            parser = Struct('formatted_entry', *fields)
            context[self.format_field + '_parser'] = parser
        return parser._parse(stream, context)