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
class SwiftDeleteObject:
    """
    Class for specifying an object delete, allowing the headers/metadata to be
    specified separately for each individual object.
    """

    def __init__(self, object_name, options=None):
        if not (isinstance(object_name, str) and object_name):
            raise SwiftError('Object names must be specified as non-empty strings')
        self.object_name = object_name
        self.options = options