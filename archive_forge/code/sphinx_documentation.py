import os
import traceback
import warnings
from os.path import join
from stat import ST_MTIME
import re
import runpy
from docutils import nodes
from docutils.parsers.rst.roles import set_classes
from subprocess import check_call, DEVNULL, CalledProcessError
from pathlib import Path
import matplotlib
Manually inspect generated files.