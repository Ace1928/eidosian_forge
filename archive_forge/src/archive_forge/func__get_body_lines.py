import collections
from llvmlite.ir import context, values, types, _utils
def _get_body_lines(self):
    lines = [it.get_declaration() for it in self.get_identified_types().values()]
    lines += [str(v) for v in self.globals.values()]
    return lines