import logging
import os
from collections import defaultdict
from concurrent.futures import as_completed, CancelledError, TimeoutError
from copy import deepcopy
from errno import EEXIST, ENOENT
from hashlib import md5
from io import StringIO
from os import environ, makedirs, stat, utime
from os.path import (
from posixpath import join as urljoin
from random import shuffle
from time import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
from urllib.parse import quote
import json
from swiftclient import Connection
from swiftclient.command_helpers import (
from swiftclient.utils import (
from swiftclient.exceptions import ClientException
from swiftclient.multithreading import MultiThreadingManager
class SwiftError(Exception):

    def __init__(self, value, container=None, obj=None, segment=None, exc=None):
        self.value = value
        self.container = container
        self.obj = obj
        self.segment = segment
        self.exception = exc

    def __str__(self):
        value = repr(self.value)
        if self.container is not None:
            value += ' container:%s' % self.container
        if self.obj is not None:
            value += ' object:%s' % self.obj
        if self.segment is not None:
            value += ' segment:%s' % self.segment
        return value

    def __repr__(self):
        return str(self)