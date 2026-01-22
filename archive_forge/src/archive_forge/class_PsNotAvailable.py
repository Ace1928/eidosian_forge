import errno
import subprocess
import sys
from ._core import Process
class PsNotAvailable(EnvironmentError):
    pass