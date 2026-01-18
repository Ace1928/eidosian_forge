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
def xmlinfo(self, id):
    """Return the XML info record for the given item"""
    self._update_index()
    for package in self._index.findall('packages/package'):
        if package.get('id') == id:
            return package
    for collection in self._index.findall('collections/collection'):
        if collection.get('id') == id:
            return collection
    raise ValueError('Package %r not found in index' % id)