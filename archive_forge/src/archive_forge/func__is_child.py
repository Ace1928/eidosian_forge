import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def _is_child(self, path):
    return posixpath.dirname(path.at.rstrip('/')) == self.at.rstrip('/')