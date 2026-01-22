from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from .util import _mapper_property_as_plain_name
from .. import exc as sa_exc
from .. import util
from ..exc import MultipleResultsFound  # noqa
from ..exc import NoResultFound  # noqa
class UnmappedClassError(UnmappedError):
    """An mapping operation was requested for an unknown class."""

    def __init__(self, cls: Type[_T], msg: Optional[str]=None):
        if not msg:
            msg = _default_unmapped(cls)
        UnmappedError.__init__(self, msg)

    def __reduce__(self) -> Any:
        return (self.__class__, (None, self.args[0]))