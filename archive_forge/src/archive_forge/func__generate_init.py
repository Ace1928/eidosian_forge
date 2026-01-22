from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import base
from . import collections
from . import exc
from . import interfaces
from . import state
from ._typing import _O
from .attributes import _is_collection_attribute_impl
from .. import util
from ..event import EventTarget
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def _generate_init(class_, class_manager, original_init):
    """Build an __init__ decorator that triggers ClassManager events."""
    if original_init is None:
        original_init = class_.__init__
    func_body = 'def __init__(%(apply_pos)s):\n    new_state = class_manager._new_state_if_none(%(self_arg)s)\n    if new_state:\n        return new_state._initialize_instance(%(apply_kw)s)\n    else:\n        return original_init(%(apply_kw)s)\n'
    func_vars = util.format_argspec_init(original_init, grouped=False)
    func_text = func_body % func_vars
    func_defaults = getattr(original_init, '__defaults__', None)
    func_kw_defaults = getattr(original_init, '__kwdefaults__', None)
    env = locals().copy()
    env['__name__'] = __name__
    exec(func_text, env)
    __init__ = env['__init__']
    __init__.__doc__ = original_init.__doc__
    __init__._sa_original_init = original_init
    if func_defaults:
        __init__.__defaults__ = func_defaults
    if func_kw_defaults:
        __init__.__kwdefaults__ = func_kw_defaults
    return __init__