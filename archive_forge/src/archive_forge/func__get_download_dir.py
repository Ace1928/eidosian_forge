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
def _get_download_dir(self):
    """
        The default directory to which packages will be downloaded.
        This defaults to the value returned by ``default_download_dir()``.
        To override this default on a case-by-case basis, use the
        ``download_dir`` argument when calling ``download()``.
        """
    return self._download_dir