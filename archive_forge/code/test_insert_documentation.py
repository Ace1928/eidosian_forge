from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
test #9739, #9808 (similar to #9701).

        this tests insertmanyvalues in conjunction with various datatypes.

        These tests are particularly for the asyncpg driver which needs
        most types to be explicitly cast for the new IMV format

        