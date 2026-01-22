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
class SwiftUploadObject:
    """
    Class for specifying an object upload, allowing the object source, name and
    options to be specified separately for each individual object.
    """

    def __init__(self, source, object_name=None, options=None):
        if isinstance(source, str):
            self.object_name = object_name or source
        elif source is None or hasattr(source, 'read'):
            if not object_name or not isinstance(object_name, str):
                raise SwiftError('Object names must be specified as strings for uploads from None or file like objects.')
            self.object_name = object_name
        else:
            raise SwiftError('Unexpected source type for SwiftUploadObject: {0}'.format(type(source)))
        if not self.object_name:
            raise SwiftError('Object names must not be empty strings')
        self.object_name = self.object_name.lstrip('/')
        self.options = options
        self.source = source