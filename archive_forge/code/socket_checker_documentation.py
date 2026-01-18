from __future__ import annotations
import errno
import select
import sys
from typing import Any, Optional, cast
Return True if we know socket has been closed, False otherwise.