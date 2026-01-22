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
def _info_edit(self, info_key):
    self._info_save()
    entry, callback = self._info[info_key]
    entry['state'] = 'normal'
    entry['relief'] = 'sunken'
    entry.focus()