import sys
import _winapi
import itertools
import msvcrt
import os
import subprocess
import tempfile
import warnings
Replacement for subprocess.Popen using overlapped pipe handles.

    The stdin, stdout, stderr are None or instances of PipeHandle.
    