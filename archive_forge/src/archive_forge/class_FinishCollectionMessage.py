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
class FinishCollectionMessage(DownloaderMessage):
    """Data server has finished working on a collection of packages."""

    def __init__(self, collection):
        self.collection = collection