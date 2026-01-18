import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion

        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.
        