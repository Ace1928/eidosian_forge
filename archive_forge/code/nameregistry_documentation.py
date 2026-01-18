import threading
from typing import Any
from typing import Callable
from typing import MutableMapping
import weakref
Get and possibly create the value.

        :param identifier: Hash key for the value.
         If the creation function is called, this identifier
         will also be passed to the creation function.
        :param \*args, \**kw: Additional arguments which will
         also be passed to the creation function if it is
         called.

        