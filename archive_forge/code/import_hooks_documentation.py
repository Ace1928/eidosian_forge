import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
Implements a post-import hook mechanism.

Styled as per PEP-369. Note that it doesn't cope with modules being reloaded.

Note: This file is based on
https://github.com/GrahamDumpleton/wrapt/blob/1.12.1/src/wrapt/importer.py
and manual backports of later patches up to 1.15.0 in the wrapt repository
(with slight modifications).
