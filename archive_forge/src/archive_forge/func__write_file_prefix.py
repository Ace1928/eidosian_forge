import contextlib
import os
import pathlib
import shutil
import stat
import sys
import zipfile
import {module}
def _write_file_prefix(f, interpreter):
    """Write a shebang line."""
    if interpreter:
        shebang = b'#!' + interpreter.encode(shebang_encoding) + b'\n'
        f.write(shebang)