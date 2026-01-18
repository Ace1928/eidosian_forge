import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def write_history():
    try:
        readline.write_history_file(history)
    except OSError:
        pass