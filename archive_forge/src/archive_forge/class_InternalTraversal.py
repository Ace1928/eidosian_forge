from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
class InternalTraversal(Enum):
    """Defines visitor symbols used for internal traversal.

    The :class:`.InternalTraversal` class is used in two ways.  One is that
    it can serve as the superclass for an object that implements the
    various visit methods of the class.   The other is that the symbols
    themselves of :class:`.InternalTraversal` are used within
    the ``_traverse_internals`` collection.   Such as, the :class:`.Case`
    object defines ``_traverse_internals`` as ::

        class Case(ColumnElement[_T]):
            _traverse_internals = [
                ("value", InternalTraversal.dp_clauseelement),
                ("whens", InternalTraversal.dp_clauseelement_tuples),
                ("else_", InternalTraversal.dp_clauseelement),
            ]

    Above, the :class:`.Case` class indicates its internal state as the
    attributes named ``value``, ``whens``, and ``else_``.    They each
    link to an :class:`.InternalTraversal` method which indicates the type
    of datastructure to which each attribute refers.

    Using the ``_traverse_internals`` structure, objects of type
    :class:`.InternalTraversible` will have the following methods automatically
    implemented:

    * :meth:`.HasTraverseInternals.get_children`

    * :meth:`.HasTraverseInternals._copy_internals`

    * :meth:`.HasCacheKey._gen_cache_key`

    Subclasses can also implement these methods directly, particularly for the
    :meth:`.HasTraverseInternals._copy_internals` method, when special steps
    are needed.

    .. versionadded:: 1.4

    """
    dp_has_cache_key = 'HC'
    'Visit a :class:`.HasCacheKey` object.'
    dp_has_cache_key_list = 'HL'
    'Visit a list of :class:`.HasCacheKey` objects.'
    dp_clauseelement = 'CE'
    'Visit a :class:`_expression.ClauseElement` object.'
    dp_fromclause_canonical_column_collection = 'FC'
    'Visit a :class:`_expression.FromClause` object in the context of the\n    ``columns`` attribute.\n\n    The column collection is "canonical", meaning it is the originally\n    defined location of the :class:`.ColumnClause` objects.   Right now\n    this means that the object being visited is a\n    :class:`_expression.TableClause`\n    or :class:`_schema.Table` object only.\n\n    '
    dp_clauseelement_tuples = 'CTS'
    'Visit a list of tuples which contain :class:`_expression.ClauseElement`\n    objects.\n\n    '
    dp_clauseelement_list = 'CL'
    'Visit a list of :class:`_expression.ClauseElement` objects.\n\n    '
    dp_clauseelement_tuple = 'CT'
    'Visit a tuple of :class:`_expression.ClauseElement` objects.\n\n    '
    dp_executable_options = 'EO'
    dp_with_context_options = 'WC'
    dp_fromclause_ordered_set = 'CO'
    'Visit an ordered set of :class:`_expression.FromClause` objects. '
    dp_string = 'S'
    'Visit a plain string value.\n\n    Examples include table and column names, bound parameter keys, special\n    keywords such as "UNION", "UNION ALL".\n\n    The string value is considered to be significant for cache key\n    generation.\n\n    '
    dp_string_list = 'SL'
    'Visit a list of strings.'
    dp_anon_name = 'AN'
    'Visit a potentially "anonymized" string value.\n\n    The string value is considered to be significant for cache key\n    generation.\n\n    '
    dp_boolean = 'B'
    'Visit a boolean value.\n\n    The boolean value is considered to be significant for cache key\n    generation.\n\n    '
    dp_operator = 'O'
    'Visit an operator.\n\n    The operator is a function from the :mod:`sqlalchemy.sql.operators`\n    module.\n\n    The operator value is considered to be significant for cache key\n    generation.\n\n    '
    dp_type = 'T'
    'Visit a :class:`.TypeEngine` object\n\n    The type object is considered to be significant for cache key\n    generation.\n\n    '
    dp_plain_dict = 'PD'
    'Visit a dictionary with string keys.\n\n    The keys of the dictionary should be strings, the values should\n    be immutable and hashable.   The dictionary is considered to be\n    significant for cache key generation.\n\n    '
    dp_dialect_options = 'DO'
    'Visit a dialect options structure.'
    dp_string_clauseelement_dict = 'CD'
    'Visit a dictionary of string keys to :class:`_expression.ClauseElement`\n    objects.\n\n    '
    dp_string_multi_dict = 'MD'
    'Visit a dictionary of string keys to values which may either be\n    plain immutable/hashable or :class:`.HasCacheKey` objects.\n\n    '
    dp_annotations_key = 'AK'
    'Visit the _annotations_cache_key element.\n\n    This is a dictionary of additional information about a ClauseElement\n    that modifies its role.  It should be included when comparing or caching\n    objects, however generating this key is relatively expensive.   Visitors\n    should check the "_annotations" dict for non-None first before creating\n    this key.\n\n    '
    dp_plain_obj = 'PO'
    'Visit a plain python object.\n\n    The value should be immutable and hashable, such as an integer.\n    The value is considered to be significant for cache key generation.\n\n    '
    dp_named_ddl_element = 'DD'
    'Visit a simple named DDL element.\n\n    The current object used by this method is the :class:`.Sequence`.\n\n    The object is only considered to be important for cache key generation\n    as far as its name, but not any other aspects of it.\n\n    '
    dp_prefix_sequence = 'PS'
    'Visit the sequence represented by :class:`_expression.HasPrefixes`\n    or :class:`_expression.HasSuffixes`.\n\n    '
    dp_table_hint_list = 'TH'
    'Visit the ``_hints`` collection of a :class:`_expression.Select`\n    object.\n\n    '
    dp_setup_join_tuple = 'SJ'
    dp_memoized_select_entities = 'ME'
    dp_statement_hint_list = 'SH'
    'Visit the ``_statement_hints`` collection of a\n    :class:`_expression.Select`\n    object.\n\n    '
    dp_unknown_structure = 'UK'
    'Visit an unknown structure.\n\n    '
    dp_dml_ordered_values = 'DML_OV'
    'Visit the values() ordered tuple list of an\n    :class:`_expression.Update` object.'
    dp_dml_values = 'DML_V'
    'Visit the values() dictionary of a :class:`.ValuesBase`\n    (e.g. Insert or Update) object.\n\n    '
    dp_dml_multi_values = 'DML_MV'
    'Visit the values() multi-valued list of dictionaries of an\n    :class:`_expression.Insert` object.\n\n    '
    dp_propagate_attrs = 'PA'
    'Visit the propagate attrs dict.  This hardcodes to the particular\n    elements we care about right now.'
    'Symbols that follow are additional symbols that are useful in\n    caching applications.\n\n    Traversals for :class:`_expression.ClauseElement` objects only need to use\n    those symbols present in :class:`.InternalTraversal`.  However, for\n    additional caching use cases within the ORM, symbols dealing with the\n    :class:`.HasCacheKey` class are added here.\n\n    '
    dp_ignore = 'IG'
    'Specify an object that should be ignored entirely.\n\n    This currently applies function call argument caching where some\n    arguments should not be considered to be part of a cache key.\n\n    '
    dp_inspectable = 'IS'
    'Visit an inspectable object where the return value is a\n    :class:`.HasCacheKey` object.'
    dp_multi = 'M'
    'Visit an object that may be a :class:`.HasCacheKey` or may be a\n    plain hashable object.'
    dp_multi_list = 'MT'
    'Visit a tuple containing elements that may be :class:`.HasCacheKey` or\n    may be a plain hashable object.'
    dp_has_cache_key_tuples = 'HT'
    'Visit a list of tuples which contain :class:`.HasCacheKey`\n    objects.\n\n    '
    dp_inspectable_list = 'IL'
    'Visit a list of inspectable objects which upon inspection are\n    HasCacheKey objects.'