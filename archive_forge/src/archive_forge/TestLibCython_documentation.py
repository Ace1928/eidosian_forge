import os
import re
import sys
import shutil
import warnings
import textwrap
import unittest
import tempfile
import subprocess
from distutils import ccompiler
import runtests
import Cython.Distutils.extension
import Cython.Distutils.old_build_ext as build_ext
from Cython.Debugger import Cygdb as cygdb

        Run gdb and have cygdb import the debug information from the code
        defined in TestParseTreeTransforms's setUp method
        