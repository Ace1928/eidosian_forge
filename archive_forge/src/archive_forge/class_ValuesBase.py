from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard
class ValuesBase(UpdateBase):
    """Supplies support for :meth:`.ValuesBase.values` to
    INSERT and UPDATE constructs."""
    __visit_name__ = 'values_base'
    _supports_multi_parameters = False
    select: Optional[Select[Any]] = None
    'SELECT statement for INSERT .. FROM SELECT'
    _post_values_clause: Optional[ClauseElement] = None
    'used by extensions to Insert etc. to add additional syntacitcal\n    constructs, e.g. ON CONFLICT etc.'
    _values: Optional[util.immutabledict[_DMLColumnElement, Any]] = None
    _multi_values: Tuple[Union[Sequence[Dict[_DMLColumnElement, Any]], Sequence[Sequence[Any]]], ...] = ()
    _ordered_values: Optional[List[Tuple[_DMLColumnElement, Any]]] = None
    _select_names: Optional[List[str]] = None
    _inline: bool = False

    def __init__(self, table: _DMLTableArgument):
        self.table = coercions.expect(roles.DMLTableRole, table, apply_propagate_attrs=self)

    @_generative
    @_exclusive_against('_select_names', '_ordered_values', msgs={'_select_names': 'This construct already inserts from a SELECT', '_ordered_values': 'This statement already has ordered values present'})
    def values(self, *args: Union[_DMLColumnKeyMapping[Any], Sequence[Any]], **kwargs: Any) -> Self:
        """Specify a fixed VALUES clause for an INSERT statement, or the SET
        clause for an UPDATE.

        Note that the :class:`_expression.Insert` and
        :class:`_expression.Update`
        constructs support
        per-execution time formatting of the VALUES and/or SET clauses,
        based on the arguments passed to :meth:`_engine.Connection.execute`.
        However, the :meth:`.ValuesBase.values` method can be used to "fix" a
        particular set of parameters into the statement.

        Multiple calls to :meth:`.ValuesBase.values` will produce a new
        construct, each one with the parameter list modified to include
        the new parameters sent.  In the typical case of a single
        dictionary of parameters, the newly passed keys will replace
        the same keys in the previous construct.  In the case of a list-based
        "multiple values" construct, each new list of values is extended
        onto the existing list of values.

        :param \\**kwargs: key value pairs representing the string key
          of a :class:`_schema.Column`
          mapped to the value to be rendered into the
          VALUES or SET clause::

                users.insert().values(name="some name")

                users.update().where(users.c.id==5).values(name="some name")

        :param \\*args: As an alternative to passing key/value parameters,
         a dictionary, tuple, or list of dictionaries or tuples can be passed
         as a single positional argument in order to form the VALUES or
         SET clause of the statement.  The forms that are accepted vary
         based on whether this is an :class:`_expression.Insert` or an
         :class:`_expression.Update` construct.

         For either an :class:`_expression.Insert` or
         :class:`_expression.Update`
         construct, a single dictionary can be passed, which works the same as
         that of the kwargs form::

            users.insert().values({"name": "some name"})

            users.update().values({"name": "some new name"})

         Also for either form but more typically for the
         :class:`_expression.Insert` construct, a tuple that contains an
         entry for every column in the table is also accepted::

            users.insert().values((5, "some name"))

         The :class:`_expression.Insert` construct also supports being
         passed a list of dictionaries or full-table-tuples, which on the
         server will render the less common SQL syntax of "multiple values" -
         this syntax is supported on backends such as SQLite, PostgreSQL,
         MySQL, but not necessarily others::

            users.insert().values([
                                {"name": "some name"},
                                {"name": "some other name"},
                                {"name": "yet another name"},
                            ])

         The above form would render a multiple VALUES statement similar to::

                INSERT INTO users (name) VALUES
                                (:name_1),
                                (:name_2),
                                (:name_3)

         It is essential to note that **passing multiple values is
         NOT the same as using traditional executemany() form**.  The above
         syntax is a **special** syntax not typically used.  To emit an
         INSERT statement against multiple rows, the normal method is
         to pass a multiple values list to the
         :meth:`_engine.Connection.execute`
         method, which is supported by all database backends and is generally
         more efficient for a very large number of parameters.

           .. seealso::

               :ref:`tutorial_multiple_parameters` - an introduction to
               the traditional Core method of multiple parameter set
               invocation for INSERTs and other statements.

          The UPDATE construct also supports rendering the SET parameters
          in a specific order.  For this feature refer to the
          :meth:`_expression.Update.ordered_values` method.

           .. seealso::

              :meth:`_expression.Update.ordered_values`


        """
        if args:
            arg = args[0]
            if kwargs:
                raise exc.ArgumentError("Can't pass positional and kwargs to values() simultaneously")
            elif len(args) > 1:
                raise exc.ArgumentError('Only a single dictionary/tuple or list of dictionaries/tuples is accepted positionally.')
            elif isinstance(arg, collections_abc.Sequence):
                if arg and isinstance(arg[0], dict):
                    multi_kv_generator = DMLState.get_plugin_class(self)._get_multi_crud_kv_pairs
                    self._multi_values += (multi_kv_generator(self, arg),)
                    return self
                if arg and isinstance(arg[0], (list, tuple)):
                    self._multi_values += (arg,)
                    return self
                if TYPE_CHECKING:
                    assert isinstance(self, Insert)
                arg = {c.key: value for c, value in zip(self.table.c, arg)}
        else:
            arg = cast('Dict[_DMLColumnArgument, Any]', kwargs)
            if args:
                raise exc.ArgumentError('Only a single dictionary/tuple or list of dictionaries/tuples is accepted positionally.')
        kv_generator = DMLState.get_plugin_class(self)._get_crud_kv_pairs
        coerced_arg = dict(kv_generator(self, arg.items(), True))
        if self._values:
            self._values = self._values.union(coerced_arg)
        else:
            self._values = util.immutabledict(coerced_arg)
        return self