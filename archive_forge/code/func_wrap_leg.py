from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from .registry import _ET
from .registry import _ListenerFnType
from .. import util
from ..util.compat import FullArgSpec
def wrap_leg(*args: Any, **kw: Any) -> Any:
    util.warn_deprecated(warning_txt, version=since)
    argdict = dict(zip(dispatch_collection.arg_names, args))
    args_from_dict = [argdict[name] for name in argnames]
    if has_kw:
        return fn(*args_from_dict, **kw)
    else:
        return fn(*args_from_dict)