from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
class AutogenContext:
    """Maintains configuration and state that's specific to an
    autogenerate operation."""
    metadata: Optional[MetaData] = None
    'The :class:`~sqlalchemy.schema.MetaData` object\n    representing the destination.\n\n    This object is the one that is passed within ``env.py``\n    to the :paramref:`.EnvironmentContext.configure.target_metadata`\n    parameter.  It represents the structure of :class:`.Table` and other\n    objects as stated in the current database model, and represents the\n    destination structure for the database being examined.\n\n    While the :class:`~sqlalchemy.schema.MetaData` object is primarily\n    known as a collection of :class:`~sqlalchemy.schema.Table` objects,\n    it also has an :attr:`~sqlalchemy.schema.MetaData.info` dictionary\n    that may be used by end-user schemes to store additional schema-level\n    objects that are to be compared in custom autogeneration schemes.\n\n    '
    connection: Optional[Connection] = None
    'The :class:`~sqlalchemy.engine.base.Connection` object currently\n    connected to the database backend being compared.\n\n    This is obtained from the :attr:`.MigrationContext.bind` and is\n    ultimately set up in the ``env.py`` script.\n\n    '
    dialect: Optional[Dialect] = None
    'The :class:`~sqlalchemy.engine.Dialect` object currently in use.\n\n    This is normally obtained from the\n    :attr:`~sqlalchemy.engine.base.Connection.dialect` attribute.\n\n    '
    imports: Set[str] = None
    'A ``set()`` which contains string Python import directives.\n\n    The directives are to be rendered into the ``${imports}`` section\n    of a script template.  The set is normally empty and can be modified\n    within hooks such as the\n    :paramref:`.EnvironmentContext.configure.render_item` hook.\n\n    .. seealso::\n\n        :ref:`autogen_render_types`\n\n    '
    migration_context: MigrationContext = None
    'The :class:`.MigrationContext` established by the ``env.py`` script.'

    def __init__(self, migration_context: MigrationContext, metadata: Optional[MetaData]=None, opts: Optional[Dict[str, Any]]=None, autogenerate: bool=True) -> None:
        if autogenerate and migration_context is not None and migration_context.as_sql:
            raise util.CommandError("autogenerate can't use as_sql=True as it prevents querying the database for schema information")
        if opts is None:
            opts = migration_context.opts
        self.metadata = metadata = opts.get('target_metadata', None) if metadata is None else metadata
        if autogenerate and metadata is None and (migration_context is not None) and (migration_context.script is not None):
            raise util.CommandError("Can't proceed with --autogenerate option; environment script %s does not provide a MetaData object or sequence of objects to the context." % migration_context.script.env_py_location)
        include_object = opts.get('include_object', None)
        include_name = opts.get('include_name', None)
        object_filters = []
        name_filters = []
        if include_object:
            object_filters.append(include_object)
        if include_name:
            name_filters.append(include_name)
        self._object_filters = object_filters
        self._name_filters = name_filters
        self.migration_context = migration_context
        if self.migration_context is not None:
            self.connection = self.migration_context.bind
            self.dialect = self.migration_context.dialect
        self.imports = set()
        self.opts: Dict[str, Any] = opts
        self._has_batch: bool = False

    @util.memoized_property
    def inspector(self) -> Inspector:
        if self.connection is None:
            raise TypeError("can't return inspector as this AutogenContext has no database connection")
        return inspect(self.connection)

    @contextlib.contextmanager
    def _within_batch(self) -> Iterator[None]:
        self._has_batch = True
        yield
        self._has_batch = False

    def run_name_filters(self, name: Optional[str], type_: NameFilterType, parent_names: NameFilterParentNames) -> bool:
        """Run the context's name filters and return True if the targets
        should be part of the autogenerate operation.

        This method should be run for every kind of name encountered within the
        reflection side of an autogenerate operation, giving the environment
        the chance to filter what names should be reflected as database
        objects.  The filters here are produced directly via the
        :paramref:`.EnvironmentContext.configure.include_name` parameter.

        """
        if 'schema_name' in parent_names:
            if type_ == 'table':
                table_name = name
            else:
                table_name = parent_names.get('table_name', None)
            if table_name:
                schema_name = parent_names['schema_name']
                if schema_name:
                    parent_names['schema_qualified_table_name'] = '%s.%s' % (schema_name, table_name)
                else:
                    parent_names['schema_qualified_table_name'] = table_name
        for fn in self._name_filters:
            if not fn(name, type_, parent_names):
                return False
        else:
            return True

    def run_object_filters(self, object_: SchemaItem, name: sqla_compat._ConstraintName, type_: NameFilterType, reflected: bool, compare_to: Optional[SchemaItem]) -> bool:
        """Run the context's object filters and return True if the targets
        should be part of the autogenerate operation.

        This method should be run for every kind of object encountered within
        an autogenerate operation, giving the environment the chance
        to filter what objects should be included in the comparison.
        The filters here are produced directly via the
        :paramref:`.EnvironmentContext.configure.include_object` parameter.

        """
        for fn in self._object_filters:
            if not fn(object_, name, type_, reflected, compare_to):
                return False
        else:
            return True
    run_filters = run_object_filters

    @util.memoized_property
    def sorted_tables(self) -> List[Table]:
        """Return an aggregate of the :attr:`.MetaData.sorted_tables`
        collection(s).

        For a sequence of :class:`.MetaData` objects, this
        concatenates the :attr:`.MetaData.sorted_tables` collection
        for each individual :class:`.MetaData`  in the order of the
        sequence.  It does **not** collate the sorted tables collections.

        """
        result = []
        for m in util.to_list(self.metadata):
            result.extend(m.sorted_tables)
        return result

    @util.memoized_property
    def table_key_to_table(self) -> Dict[str, Table]:
        """Return an aggregate  of the :attr:`.MetaData.tables` dictionaries.

        The :attr:`.MetaData.tables` collection is a dictionary of table key
        to :class:`.Table`; this method aggregates the dictionary across
        multiple :class:`.MetaData` objects into one dictionary.

        Duplicate table keys are **not** supported; if two :class:`.MetaData`
        objects contain the same table key, an exception is raised.

        """
        result: Dict[str, Table] = {}
        for m in util.to_list(self.metadata):
            intersect = set(result).intersection(set(m.tables))
            if intersect:
                raise ValueError('Duplicate table keys across multiple MetaData objects: %s' % ', '.join(('"%s"' % key for key in sorted(intersect))))
            result.update(m.tables)
        return result