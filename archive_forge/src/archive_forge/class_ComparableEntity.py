from __future__ import annotations
import sqlalchemy as sa
from .. import exc as sa_exc
from ..orm.writeonly import WriteOnlyCollection
class ComparableEntity(ComparableMixin, BasicEntity):

    def __hash__(self):
        return hash(self.__class__)