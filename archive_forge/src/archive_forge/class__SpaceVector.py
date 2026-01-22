from .array import ARRAY
from .types import OID
from .types import REGCLASS
from ... import Column
from ... import func
from ... import MetaData
from ... import Table
from ...types import BigInteger
from ...types import Boolean
from ...types import CHAR
from ...types import Float
from ...types import Integer
from ...types import SmallInteger
from ...types import String
from ...types import Text
from ...types import TypeDecorator
class _SpaceVector:

    def result_processor(self, dialect, coltype):

        def process(value):
            if value is None:
                return value
            return [int(p) for p in value.split(' ')]
        return process