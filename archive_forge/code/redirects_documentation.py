import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial

    Redirect ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file in the path specified by ``to_file``.

    This method redirects the underlying std file descriptor (not just python's ``sys.stdout|stderr``).
    See usage for details.

    Directory of ``dst_filename`` is assumed to exist and the destination file
    is overwritten if it already exists.

    .. note:: Due to buffering cross source writes are not guaranteed to
              appear in wall-clock order. For instance in the example below
              it is possible for the C-outputs to appear before the python
              outputs in the log file.

    Usage:

    ::

     # syntactic-sugar for redirect("stdout", "tmp/stdout.log")
     with redirect_stdout("/tmp/stdout.log"):
        print("python stdouts are redirected")
        libc = ctypes.CDLL("libc.so.6")
        libc.printf(b"c stdouts are also redirected"
        os.system("echo system stdouts are also redirected")

     print("stdout restored")

    