import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_metadata_struct(self):
    self.printer.source_map[self.printer.lineno] = max(self.printer.source_map)
    struct = {'filename': self.compiler.filename, 'uri': self.compiler.uri, 'source_encoding': self.compiler.source_encoding, 'line_map': self.printer.source_map}
    self.printer.writelines('"""', '__M_BEGIN_METADATA', json.dumps(struct), '__M_END_METADATA\n"""')