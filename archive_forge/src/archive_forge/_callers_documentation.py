from __future__ import annotations
import warnings
from typing import cast
from typing import Generator
from typing import Mapping
from typing import NoReturn
from typing import Sequence
from typing import Tuple
from typing import Union
from ._hooks import HookImpl
from ._result import HookCallError
from ._result import Result
from ._warnings import PluggyTeardownRaisedWarning
Execute a call into multiple python functions/methods and return the
    result(s).

    ``caller_kwargs`` comes from HookCaller.__call__().
    