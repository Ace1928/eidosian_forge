from ... import cast
from ... import Column
from ... import MetaData
from ... import Table
from ...ext.compiler import compiles
from ...sql import expression
from ...types import Boolean
from ...types import Integer
from ...types import Numeric
from ...types import NVARCHAR
from ...types import String
from ...types import TypeDecorator
from ...types import Unicode
class CoerceUnicode(TypeDecorator):
    impl = Unicode
    cache_ok = True

    def bind_expression(self, bindvalue):
        return _cast_on_2005(bindvalue)