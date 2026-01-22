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
def _relationships_for_fks(automap_base: Type[Any], map_config: _DeferredMapperConfig, table_to_map_config: Union[Dict[Optional[Table], _DeferredMapperConfig], Dict[Table, _DeferredMapperConfig]], collection_class: type, name_for_scalar_relationship: NameForScalarRelationshipType, name_for_collection_relationship: NameForCollectionRelationshipType, generate_relationship: GenerateRelationshipType) -> None:
    local_table = cast('Optional[Table]', map_config.local_table)
    local_cls = cast('Optional[Type[Any]]', map_config.cls)
    if local_table is None or local_cls is None:
        return
    for constraint in local_table.constraints:
        if isinstance(constraint, ForeignKeyConstraint):
            fks = constraint.elements
            referred_table = fks[0].column.table
            referred_cfg = table_to_map_config.get(referred_table, None)
            if referred_cfg is None:
                continue
            referred_cls = referred_cfg.cls
            if local_cls is not referred_cls and issubclass(local_cls, referred_cls):
                continue
            relationship_name = name_for_scalar_relationship(automap_base, local_cls, referred_cls, constraint)
            backref_name = name_for_collection_relationship(automap_base, referred_cls, local_cls, constraint)
            o2m_kws: Dict[str, Union[str, bool]] = {}
            nullable = False not in {fk.parent.nullable for fk in fks}
            if not nullable:
                o2m_kws['cascade'] = 'all, delete-orphan'
                if constraint.ondelete and constraint.ondelete.lower() == 'cascade':
                    o2m_kws['passive_deletes'] = True
            elif constraint.ondelete and constraint.ondelete.lower() == 'set null':
                o2m_kws['passive_deletes'] = True
            create_backref = backref_name not in referred_cfg.properties
            if relationship_name not in map_config.properties:
                if create_backref:
                    backref_obj = generate_relationship(automap_base, interfaces.ONETOMANY, backref, backref_name, referred_cls, local_cls, collection_class=collection_class, **o2m_kws)
                else:
                    backref_obj = None
                rel = generate_relationship(automap_base, interfaces.MANYTOONE, relationship, relationship_name, local_cls, referred_cls, foreign_keys=[fk.parent for fk in constraint.elements], backref=backref_obj, remote_side=[fk.column for fk in constraint.elements])
                if rel is not None:
                    map_config.properties[relationship_name] = rel
                    if not create_backref:
                        referred_cfg.properties[backref_name].back_populates = relationship_name
            elif create_backref:
                rel = generate_relationship(automap_base, interfaces.ONETOMANY, relationship, backref_name, referred_cls, local_cls, foreign_keys=[fk.parent for fk in constraint.elements], back_populates=relationship_name, collection_class=collection_class, **o2m_kws)
                if rel is not None:
                    referred_cfg.properties[backref_name] = rel
                    map_config.properties[relationship_name].back_populates = backref_name