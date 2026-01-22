from ``Engineer`` to ``Employee``, we need to set up both the relationship
from __future__ import annotations
import dataclasses
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import backref
from ..orm import declarative_base as _declarative_base
from ..orm import exc as orm_exc
from ..orm import interfaces
from ..orm import relationship
from ..orm.decl_base import _DeferredMapperConfig
from ..orm.mapper import _CONFIGURE_MUTEX
from ..schema import ForeignKeyConstraint
from ..sql import and_
from ..util import Properties
from ..util.typing import Protocol
class AutomapBase:
    """Base class for an "automap" schema.

    The :class:`.AutomapBase` class can be compared to the "declarative base"
    class that is produced by the :func:`.declarative.declarative_base`
    function.  In practice, the :class:`.AutomapBase` class is always used
    as a mixin along with an actual declarative base.

    A new subclassable :class:`.AutomapBase` is typically instantiated
    using the :func:`.automap_base` function.

    .. seealso::

        :ref:`automap_toplevel`

    """
    __abstract__ = True
    classes: ClassVar[Properties[Type[Any]]]
    'An instance of :class:`.util.Properties` containing classes.\n\n    This object behaves much like the ``.c`` collection on a table.  Classes\n    are present under the name they were given, e.g.::\n\n        Base = automap_base()\n        Base.prepare(autoload_with=some_engine)\n\n        User, Address = Base.classes.User, Base.classes.Address\n\n    For class names that overlap with a method name of\n    :class:`.util.Properties`, such as ``items()``, the getitem form\n    is also supported::\n\n        Item = Base.classes["items"]\n\n    '
    by_module: ClassVar[ByModuleProperties]
    'An instance of :class:`.util.Properties` containing a hierarchal\n    structure of dot-separated module names linked to classes.\n\n    This collection is an alternative to the :attr:`.AutomapBase.classes`\n    collection that is useful when making use of the\n    :paramref:`.AutomapBase.prepare.modulename_for_table` parameter, which will\n    apply distinct ``__module__`` attributes to generated classes.\n\n    The default ``__module__`` an automap-generated class is\n    ``sqlalchemy.ext.automap``; to access this namespace using\n    :attr:`.AutomapBase.by_module` looks like::\n\n        User = Base.by_module.sqlalchemy.ext.automap.User\n\n    If a class had a ``__module__`` of ``mymodule.account``, accessing\n    this namespace looks like::\n\n        MyClass = Base.by_module.mymodule.account.MyClass\n\n    .. versionadded:: 2.0\n\n    .. seealso::\n\n        :ref:`automap_by_module`\n\n    '
    metadata: ClassVar[MetaData]
    'Refers to the :class:`_schema.MetaData` collection that will be used\n    for new :class:`_schema.Table` objects.\n\n    .. seealso::\n\n        :ref:`orm_declarative_metadata`\n\n    '
    _sa_automapbase_bookkeeping: ClassVar[_Bookkeeping]

    @classmethod
    @util.deprecated_params(engine=('2.0', 'The :paramref:`_automap.AutomapBase.prepare.engine` parameter is deprecated and will be removed in a future release.  Please use the :paramref:`_automap.AutomapBase.prepare.autoload_with` parameter.'), reflect=('2.0', 'The :paramref:`_automap.AutomapBase.prepare.reflect` parameter is deprecated and will be removed in a future release.  Reflection is enabled when :paramref:`_automap.AutomapBase.prepare.autoload_with` is passed.'))
    def prepare(cls: Type[AutomapBase], autoload_with: Optional[Engine]=None, engine: Optional[Any]=None, reflect: bool=False, schema: Optional[str]=None, classname_for_table: Optional[PythonNameForTableType]=None, modulename_for_table: Optional[PythonNameForTableType]=None, collection_class: Optional[Any]=None, name_for_scalar_relationship: Optional[NameForScalarRelationshipType]=None, name_for_collection_relationship: Optional[NameForCollectionRelationshipType]=None, generate_relationship: Optional[GenerateRelationshipType]=None, reflection_options: Union[Dict[_KT, _VT], immutabledict[_KT, _VT]]=util.EMPTY_DICT) -> None:
        """Extract mapped classes and relationships from the
        :class:`_schema.MetaData` and perform mappings.

        For full documentation and examples see
        :ref:`automap_basic_use`.

        :param autoload_with: an :class:`_engine.Engine` or
         :class:`_engine.Connection` with which
         to perform schema reflection; when specified, the
         :meth:`_schema.MetaData.reflect` method will be invoked within
         the scope of this method.

        :param engine: legacy; use :paramref:`.AutomapBase.autoload_with`.
         Used to indicate the :class:`_engine.Engine` or
         :class:`_engine.Connection` with which to reflect tables with,
         if :paramref:`.AutomapBase.reflect` is True.

        :param reflect: legacy; use :paramref:`.AutomapBase.autoload_with`.
         Indicates that :meth:`_schema.MetaData.reflect` should be invoked.

        :param classname_for_table: callable function which will be used to
         produce new class names, given a table name.  Defaults to
         :func:`.classname_for_table`.

        :param modulename_for_table: callable function which will be used to
         produce the effective ``__module__`` for an internally generated
         class, to allow for multiple classes of the same name in a single
         automap base which would be in different "modules".

         Defaults to ``None``, which will indicate that ``__module__`` will not
         be set explicitly; the Python runtime will use the value
         ``sqlalchemy.ext.automap`` for these classes.

         When assigning ``__module__`` to generated classes, they can be
         accessed based on dot-separated module names using the
         :attr:`.AutomapBase.by_module` collection.   Classes that have
         an explicit ``__module_`` assigned using this hook do **not** get
         placed into the :attr:`.AutomapBase.classes` collection, only
         into :attr:`.AutomapBase.by_module`.

         .. versionadded:: 2.0

         .. seealso::

            :ref:`automap_by_module`

        :param name_for_scalar_relationship: callable function which will be
         used to produce relationship names for scalar relationships.  Defaults
         to :func:`.name_for_scalar_relationship`.

        :param name_for_collection_relationship: callable function which will
         be used to produce relationship names for collection-oriented
         relationships.  Defaults to :func:`.name_for_collection_relationship`.

        :param generate_relationship: callable function which will be used to
         actually generate :func:`_orm.relationship` and :func:`.backref`
         constructs.  Defaults to :func:`.generate_relationship`.

        :param collection_class: the Python collection class that will be used
         when a new :func:`_orm.relationship`
         object is created that represents a
         collection.  Defaults to ``list``.

        :param schema: Schema name to reflect when reflecting tables using
         the :paramref:`.AutomapBase.prepare.autoload_with` parameter. The name
         is passed to the :paramref:`_schema.MetaData.reflect.schema` parameter
         of :meth:`_schema.MetaData.reflect`. When omitted, the default schema
         in use by the database connection is used.

         .. note:: The :paramref:`.AutomapBase.prepare.schema`
            parameter supports reflection of a single schema at a time.
            In order to include tables from many schemas, use
            multiple calls to :meth:`.AutomapBase.prepare`.

            For an overview of multiple-schema automap including the use
            of additional naming conventions to resolve table name
            conflicts, see the section :ref:`automap_by_module`.

            .. versionadded:: 2.0 :meth:`.AutomapBase.prepare` supports being
               directly invoked any number of times, keeping track of tables
               that have already been processed to avoid processing them
               a second time.

        :param reflection_options: When present, this dictionary of options
         will be passed to :meth:`_schema.MetaData.reflect`
         to supply general reflection-specific options like ``only`` and/or
         dialect-specific options like ``oracle_resolve_synonyms``.

         .. versionadded:: 1.4

        """
        for mr in cls.__mro__:
            if '_sa_automapbase_bookkeeping' in mr.__dict__:
                automap_base = cast('Type[AutomapBase]', mr)
                break
        else:
            assert False, "Can't locate automap base in class hierarchy"
        glbls = globals()
        if classname_for_table is None:
            classname_for_table = glbls['classname_for_table']
        if name_for_scalar_relationship is None:
            name_for_scalar_relationship = glbls['name_for_scalar_relationship']
        if name_for_collection_relationship is None:
            name_for_collection_relationship = glbls['name_for_collection_relationship']
        if generate_relationship is None:
            generate_relationship = glbls['generate_relationship']
        if collection_class is None:
            collection_class = list
        if autoload_with:
            reflect = True
        if engine:
            autoload_with = engine
        if reflect:
            assert autoload_with
            opts = dict(schema=schema, extend_existing=True, autoload_replace=False)
            if reflection_options:
                opts.update(reflection_options)
            cls.metadata.reflect(autoload_with, **opts)
        with _CONFIGURE_MUTEX:
            table_to_map_config: Union[Dict[Optional[Table], _DeferredMapperConfig], Dict[Table, _DeferredMapperConfig]] = {cast('Table', m.local_table): m for m in _DeferredMapperConfig.classes_for_base(cls, sort=False)}
            many_to_many: List[Tuple[Table, Table, List[ForeignKeyConstraint], Table]]
            many_to_many = []
            bookkeeping = automap_base._sa_automapbase_bookkeeping
            metadata_tables = cls.metadata.tables
            for table_key in set(metadata_tables).difference(bookkeeping.table_keys):
                table = metadata_tables[table_key]
                bookkeeping.table_keys.add(table_key)
                lcl_m2m, rem_m2m, m2m_const = _is_many_to_many(cls, table)
                if lcl_m2m is not None:
                    assert rem_m2m is not None
                    assert m2m_const is not None
                    many_to_many.append((lcl_m2m, rem_m2m, m2m_const, table))
                elif not table.primary_key:
                    continue
                elif table not in table_to_map_config:
                    clsdict: Dict[str, Any] = {'__table__': table}
                    if modulename_for_table is not None:
                        new_module = modulename_for_table(cls, table.name, table)
                        if new_module is not None:
                            clsdict['__module__'] = new_module
                    else:
                        new_module = None
                    newname = classname_for_table(cls, table.name, table)
                    if new_module is None and newname in cls.classes:
                        util.warn(f"Ignoring duplicate class name '{newname}' received in automap base for table {table.key} without ``__module__`` being set; consider using the ``modulename_for_table`` hook")
                        continue
                    mapped_cls = type(newname, (automap_base,), clsdict)
                    map_config = _DeferredMapperConfig.config_for_cls(mapped_cls)
                    assert map_config.cls.__name__ == newname
                    if new_module is None:
                        cls.classes[newname] = mapped_cls
                    by_module_properties: ByModuleProperties = cls.by_module
                    for token in map_config.cls.__module__.split('.'):
                        if token not in by_module_properties:
                            by_module_properties[token] = util.Properties({})
                        props = by_module_properties[token]
                        assert isinstance(props, Properties)
                        by_module_properties = props
                    by_module_properties[map_config.cls.__name__] = mapped_cls
                    table_to_map_config[table] = map_config
            for map_config in table_to_map_config.values():
                _relationships_for_fks(automap_base, map_config, table_to_map_config, collection_class, name_for_scalar_relationship, name_for_collection_relationship, generate_relationship)
            for lcl_m2m, rem_m2m, m2m_const, table in many_to_many:
                _m2m_relationship(automap_base, lcl_m2m, rem_m2m, m2m_const, table, table_to_map_config, collection_class, name_for_scalar_relationship, name_for_collection_relationship, generate_relationship)
            for map_config in _DeferredMapperConfig.classes_for_base(automap_base):
                map_config.map()
    _sa_decl_prepare = True
    "Indicate that the mapping of classes should be deferred.\n\n    The presence of this attribute name indicates to declarative\n    that the call to mapper() should not occur immediately; instead,\n    information about the table and attributes to be mapped are gathered\n    into an internal structure called _DeferredMapperConfig.  These\n    objects can be collected later using classes_for_base(), additional\n    mapping decisions can be made, and then the map() method will actually\n    apply the mapping.\n\n    The only real reason this deferral of the whole\n    thing is needed is to support primary key columns that aren't reflected\n    yet when the class is declared; everything else can theoretically be\n    added to the mapper later.  However, the _DeferredMapperConfig is a\n    nice interface in any case which exists at that not usually exposed point\n    at which declarative has the class and the Table but hasn't called\n    mapper() yet.\n\n    "

    @classmethod
    def _sa_raise_deferred_config(cls) -> NoReturn:
        raise orm_exc.UnmappedClassError(cls, msg='Class %s is a subclass of AutomapBase.  Mappings are not produced until the .prepare() method is called on the class hierarchy.' % orm_exc._safe_cls_name(cls))