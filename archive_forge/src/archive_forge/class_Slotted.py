from abc import ABC
from ..coretypes import (
from ..dispatch import dispatch
class Slotted(ABC):

    @classmethod
    def __subclasshook__(cls, subcls):
        return hasattr(subcls, '__slots__')