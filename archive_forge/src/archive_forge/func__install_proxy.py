from __future__ import annotations
import collections
from collections.abc import Iterable
import textwrap
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import uuid
import warnings
from sqlalchemy.util import asbool as asbool  # noqa: F401
from sqlalchemy.util import immutabledict as immutabledict  # noqa: F401
from sqlalchemy.util import to_list as to_list  # noqa: F401
from sqlalchemy.util import unique_list as unique_list
from .compat import inspect_getfullargspec
def _install_proxy(self) -> None:
    attr_names, modules = self._setups[self.__class__]
    for globals_, locals_ in modules:
        globals_['_proxy'] = self
        for attr_name in attr_names:
            globals_[attr_name] = getattr(self, attr_name)