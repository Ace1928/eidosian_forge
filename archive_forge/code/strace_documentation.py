import os
import signal
import subprocess
import tempfile
from . import errors
Create a StraceResult.

        :param raw_log: The output that strace created.
        