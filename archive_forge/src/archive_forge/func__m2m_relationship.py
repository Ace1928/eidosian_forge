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
def _m2m_relationship(automap_base: Type[Any], lcl_m2m: Table, rem_m2m: Table, m2m_const: List[ForeignKeyConstraint], table: Table, table_to_map_config: Union[Dict[Optional[Table], _DeferredMapperConfig], Dict[Table, _DeferredMapperConfig]], collection_class: type, name_for_scalar_relationship: NameForCollectionRelationshipType, name_for_collection_relationship: NameForCollectionRelationshipType, generate_relationship: GenerateRelationshipType) -> None:
    map_config = table_to_map_config.get(lcl_m2m, None)
    referred_cfg = table_to_map_config.get(rem_m2m, None)
    if map_config is None or referred_cfg is None:
        return
    local_cls = map_config.cls
    referred_cls = referred_cfg.cls
    relationship_name = name_for_collection_relationship(automap_base, local_cls, referred_cls, m2m_const[0])
    backref_name = name_for_collection_relationship(automap_base, referred_cls, local_cls, m2m_const[1])
    create_backref = backref_name not in referred_cfg.properties
    if table in table_to_map_config:
        overlaps = '__*'
    else:
        overlaps = None
    if relationship_name not in map_config.properties:
        if create_backref:
            backref_obj = generate_relationship(automap_base, interfaces.MANYTOMANY, backref, backref_name, referred_cls, local_cls, collection_class=collection_class, overlaps=overlaps)
        else:
            backref_obj = None
        rel = generate_relationship(automap_base, interfaces.MANYTOMANY, relationship, relationship_name, local_cls, referred_cls, overlaps=overlaps, secondary=table, primaryjoin=and_((fk.column == fk.parent for fk in m2m_const[0].elements)), secondaryjoin=and_((fk.column == fk.parent for fk in m2m_const[1].elements)), backref=backref_obj, collection_class=collection_class)
        if rel is not None:
            map_config.properties[relationship_name] = rel
            if not create_backref:
                referred_cfg.properties[backref_name].back_populates = relationship_name
    elif create_backref:
        rel = generate_relationship(automap_base, interfaces.MANYTOMANY, relationship, backref_name, referred_cls, local_cls, overlaps=overlaps, secondary=table, primaryjoin=and_((fk.column == fk.parent for fk in m2m_const[1].elements)), secondaryjoin=and_((fk.column == fk.parent for fk in m2m_const[0].elements)), back_populates=relationship_name, collection_class=collection_class)
        if rel is not None:
            referred_cfg.properties[backref_name] = rel
            map_config.properties[relationship_name].back_populates = backref_name