from ._base import *
import operator as op
class LazyLink(Enum):
    IS = 0
    NOT = 1
    IN = 2
    HAS = 3