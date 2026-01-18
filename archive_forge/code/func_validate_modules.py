import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def validate_modules(self):
    fre = re.compile('\\s*def\\s+(p_[a-zA-Z_0-9]*)\\(')
    for module in self.modules:
        try:
            lines, linen = inspect.getsourcelines(module)
        except IOError:
            continue
        counthash = {}
        for linen, line in enumerate(lines):
            linen += 1
            m = fre.match(line)
            if m:
                name = m.group(1)
                prev = counthash.get(name)
                if not prev:
                    counthash[name] = linen
                else:
                    filename = inspect.getsourcefile(module)
                    self.log.warning('%s:%d: Function %s redefined. Previously defined on line %d', filename, linen, name, prev)