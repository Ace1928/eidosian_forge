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
def _unzip_iter(filename, root, verbose=True):
    if verbose:
        sys.stdout.write('Unzipping %s' % os.path.split(filename)[1])
        sys.stdout.flush()
    try:
        zf = zipfile.ZipFile(filename)
    except zipfile.error as e:
        yield ErrorMessage(filename, 'Error with downloaded zip file')
        return
    except Exception as e:
        yield ErrorMessage(filename, e)
        return
    zf.extractall(root)
    if verbose:
        print()