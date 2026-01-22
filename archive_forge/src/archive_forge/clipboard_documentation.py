import os
import subprocess
from IPython.core.error import TryNext
import IPython.utils.py3compat as py3compat
Get the clipboard's text under Wayland using wl-paste command.

    This requires Wayland and wl-clipboard installed and running.
    