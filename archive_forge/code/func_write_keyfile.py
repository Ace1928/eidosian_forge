import argparse
import getpass
import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.error
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import distro
from .version import VERSION
def write_keyfile(keyfile_lines, mode):
    """
    Locate key file, write lines to it
    """
    output_file = get_keyfile(parser.options.output)
    if output_file == '-':
        for line in keyfile_lines:
            if line:
                sys.stdout.write(line)
                sys.stdout.write('\n\n')
        sys.stdout.flush()
    elif assert_parent_dir(output_file):
        with open(output_file, mode) as f:
            for line in keyfile_lines:
                if line.strip():
                    f.write(line)
                    f.write('\n\n')