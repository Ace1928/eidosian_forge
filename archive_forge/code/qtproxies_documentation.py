import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
LiteralObject(*args) -> new literal class

    a literal class can be used as argument in a function call

    >>> class Foo(LiteralProxyClass): pass
    >>> str(Foo(1,2,3)) == "Foo(1,2,3)"
    