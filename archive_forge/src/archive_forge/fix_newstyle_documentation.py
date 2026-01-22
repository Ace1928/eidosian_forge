from lib2to3 import fixer_base
from lib2to3.fixer_util import LParen, RParen, Name
from libfuturize.fixer_util import touch_import_top

Fixer for "class Foo: ..." -> "class Foo(object): ..."
