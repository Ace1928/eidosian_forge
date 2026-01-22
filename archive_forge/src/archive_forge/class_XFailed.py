import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
class XFailed(Failed):
    """Raised from an explicit call to pytest.xfail()."""