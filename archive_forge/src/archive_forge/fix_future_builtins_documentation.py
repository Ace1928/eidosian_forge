from __future__ import unicode_literals
from lib2to3 import fixer_base
from lib2to3.pygram import python_symbols as syms
from lib2to3.fixer_util import Name, Call, in_special_context
from libfuturize.fixer_util import touch_import_top

For the ``future`` package.

Adds this import line::

    from builtins import XYZ

for each of the functions XYZ that is used in the module.

Adds these imports after any other imports (in an initial block of them).
