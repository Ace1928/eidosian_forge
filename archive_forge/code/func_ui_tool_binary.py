import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def ui_tool_binary(binary):
    """Return the binary of a UI tool (App bundle on macOS)."""
    if sys.platform != 'darwin':
        return binary
    name = binary[0:1].upper() + binary[1:]
    return f'{name}.app/Contents/MacOS/{name}'