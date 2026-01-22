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
def _select_tab(self, event):
    self._tab = event.widget['text'].lower()
    try:
        self._fill_table()
    except HTTPError as e:
        showerror('Error reading from server', e)
    except URLError as e:
        showerror('Error connecting to server', e.reason)