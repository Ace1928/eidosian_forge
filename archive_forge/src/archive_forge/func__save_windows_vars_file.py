from __future__ import absolute_import, division, print_function
import argparse
import gzip
import pathlib
import shutil
import subprocess
import sys
from urllib import request
from xml.etree import ElementTree
import yaml
def _save_windows_vars_file(filename, vars):
    with open(filename, 'w') as fd:
        yaml.safe_dump(vars, fd)