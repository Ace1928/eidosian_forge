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
def _next_tab(self, *e):
    for i, tab in enumerate(self._tab_names):
        if tab.lower() == self._tab and i < len(self._tabs) - 1:
            self._tab = self._tab_names[i + 1].lower()
            try:
                return self._fill_table()
            except HTTPError as e:
                showerror('Error reading from server', e)
            except URLError as e:
                showerror('Error connecting to server', e.reason)