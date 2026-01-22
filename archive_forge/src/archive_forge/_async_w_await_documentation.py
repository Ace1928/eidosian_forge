import asyncio
import errno
import signal
from sys import version_info as py_version_info
from pexpect import EOF
Implementation of coroutines using ``async def``/``await`` keywords.

These keywords replaced ``@asyncio.coroutine`` and ``yield from`` from
Python 3.5 onwards.
