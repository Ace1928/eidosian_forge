from types import FunctionType, MethodType
import warnings
from .constants import (
from .ctrait import CTrait
from .trait_errors import TraitError
from .trait_base import (
from .trait_converters import (
from .trait_handler import TraitHandler
from .trait_type import (
from .trait_handlers import (
from .trait_factory import (
from .util.deprecated import deprecated
class _InstanceArgs(object):

    def __init__(self, factory, args, kw):
        self.args = (factory,) + args
        self.kw = kw