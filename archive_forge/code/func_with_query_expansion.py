from ... import exc
from ... import util
from ...sql import coercions
from ...sql import elements
from ...sql import operators
from ...sql import roles
from ...sql.base import _generative
from ...sql.base import Generative
from ...util.typing import Self
@_generative
def with_query_expansion(self) -> Self:
    """Apply the "WITH QUERY EXPANSION" modifier to the MATCH expression.

        :return: a new :class:`_mysql.match` instance with modifications
         applied.
        """
    self.modifiers = self.modifiers.union({'mysql_query_expansion': True})
    return self