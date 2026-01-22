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
class SwiftCopyObject:
    """
    Class for specifying an object copy,
    allowing the destination/headers/metadata/fresh_metadata to be specified
    separately for each individual object.
    destination and fresh_metadata should be set in options
    """

    def __init__(self, object_name, options=None):
        if not (isinstance(object_name, str) and object_name):
            raise SwiftError('Object names must be specified as non-empty strings')
        self.object_name = object_name
        self.options = options
        if self.options is None:
            self.destination = None
            self.fresh_metadata = False
        else:
            self.destination = self.options.get('destination')
            self.fresh_metadata = self.options.get('fresh_metadata', False)
        if self.destination is not None:
            destination_components = self.destination.split('/')
            if destination_components[0] or len(destination_components) < 2:
                raise SwiftError('destination must be in format /cont[/obj]')
            if not destination_components[-1]:
                raise SwiftError('destination must not end in a slash')
            if len(destination_components) == 2:
                self.destination = '{0}/{1}'.format(self.destination, object_name)