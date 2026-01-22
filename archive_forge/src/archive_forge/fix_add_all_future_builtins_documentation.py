from __future__ import unicode_literals
from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top

For the ``future`` package.

Adds this import line::

    from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                          int, list, map, next, object, oct, open, pow,
                          range, round, str, super, zip)

to a module, irrespective of whether each definition is used.

Adds these imports after any other imports (in an initial block of them).
