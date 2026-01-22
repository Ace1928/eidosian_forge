import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _disassemble(self, lineno_width=3, mark_as_current=False, offset_width=4):
    """Format instruction details for inclusion in disassembly output

        *lineno_width* sets the width of the line number field (0 omits it)
        *mark_as_current* inserts a '-->' marker arrow as part of the line
        *offset_width* sets the width of the instruction offset field
        """
    fields = []
    if lineno_width:
        if self.starts_line is not None:
            lineno_fmt = '%%%dd' % lineno_width
            fields.append(lineno_fmt % self.starts_line)
        else:
            fields.append(' ' * lineno_width)
    if mark_as_current:
        fields.append('-->')
    else:
        fields.append('   ')
    if self.is_jump_target:
        fields.append('>>')
    else:
        fields.append('  ')
    fields.append(repr(self.offset).rjust(offset_width))
    fields.append(self.opname.ljust(_OPNAME_WIDTH))
    if self.arg is not None:
        fields.append(repr(self.arg).rjust(_OPARG_WIDTH))
        if self.argrepr:
            fields.append('(' + self.argrepr + ')')
    return ' '.join(fields).rstrip()