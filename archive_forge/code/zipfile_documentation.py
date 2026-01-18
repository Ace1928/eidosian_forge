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

        Open this entry as text or binary following the semantics
        of ``pathlib.Path.open()`` by passing arguments through
        to io.TextIOWrapper().
        