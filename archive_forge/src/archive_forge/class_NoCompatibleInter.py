from typing import Generic, List, Type, TypeVar
from .errors import BzrError
from .lock import LogicalLockResult
from .pyutils import get_named_object
class NoCompatibleInter(BzrError):
    _fmt = 'No compatible object available for operations from %(source)r to %(target)r.'

    def __init__(self, source, target):
        self.source = source
        self.target = target