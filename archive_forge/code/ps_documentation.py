import errno
import subprocess
import sys
from ._core import Process
Try to look up the process tree via the output of `ps`.