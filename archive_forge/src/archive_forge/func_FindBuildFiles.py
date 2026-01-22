import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def FindBuildFiles():
    extension = '.gyp'
    files = os.listdir(os.getcwd())
    build_files = []
    for file in files:
        if file.endswith(extension):
            build_files.append(file)
    return build_files