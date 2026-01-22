from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top

Fixer that adds ``from builtins import object`` if there is a line
like this:
    class Foo(object):
