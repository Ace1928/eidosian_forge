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
def _update_table_status(self):
    for row_num in range(len(self._table)):
        status = self._ds.status(self._table[row_num, 'Identifier'])
        self._table[row_num, 'Status'] = status
    self._color_table()