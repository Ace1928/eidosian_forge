import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def _svn_revision(filename):
    """
    Helper for ``build_index()``: Calculate the subversion revision
    number for a given file (by using ``subprocess`` to run ``svn``).
    """
    p = subprocess.Popen(['svn', 'status', '-v', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0 or stderr or (not stdout):
        raise ValueError('Error determining svn_revision for %s: %s' % (os.path.split(filename)[1], textwrap.fill(stderr)))
    return stdout.split()[2]