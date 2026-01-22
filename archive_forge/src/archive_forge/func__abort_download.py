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
def _abort_download(self):
    if self._downloading:
        self._download_lock.acquire()
        self._download_abort_queue.append('abort')
        self._download_lock.release()