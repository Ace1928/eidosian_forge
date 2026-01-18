import signal
from types import FrameType
from typing import Any, Callable, Dict, Optional, Union
Install the given function as a signal handler for all common shutdown
    signals (such as SIGINT, SIGTERM, etc). If ``override_sigint`` is ``False`` the
    SIGINT handler won't be installed if there is already a handler in place
    (e.g. Pdb)
    