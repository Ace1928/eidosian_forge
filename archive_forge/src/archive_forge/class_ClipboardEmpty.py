import os
import subprocess
from IPython.core.error import TryNext
import IPython.utils.py3compat as py3compat
class ClipboardEmpty(ValueError):
    pass