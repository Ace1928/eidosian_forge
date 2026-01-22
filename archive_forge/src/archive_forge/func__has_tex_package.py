from pathlib import Path
from tempfile import TemporaryDirectory
import locale
import logging
import os
import subprocess
import sys
import matplotlib as mpl
from matplotlib import _api
def _has_tex_package(package):
    try:
        mpl.dviread.find_tex_file(f'{package}.sty')
        return True
    except FileNotFoundError:
        return False