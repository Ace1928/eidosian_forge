from __future__ import unicode_literals
from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top

Fixer for the execfile() function on Py2, which was removed in Py3.

The Lib/lib2to3/fixes/fix_execfile.py module has some problems: see
python-future issue #37. This fixer merely imports execfile() from
past.builtins and leaves the code alone.

Adds this import line::

    from past.builtins import execfile

for the function execfile() that was removed from Py3.
