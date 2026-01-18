import shutil
import subprocess
import sys
import pytest
from numpy.distutils import mingw32ccompiler
Test the mingw32ccompiler.build_import_library, which builds a
    `python.a` from the MSVC `python.lib`
    