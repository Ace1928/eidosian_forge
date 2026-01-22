import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def DiffTestStdout(golden):
    _DiffTestOutput(sys.stdout, golden)