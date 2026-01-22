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
def _info_or_id(self, info_or_id):
    if isinstance(info_or_id, str):
        return self.info(info_or_id)
    else:
        return info_or_id