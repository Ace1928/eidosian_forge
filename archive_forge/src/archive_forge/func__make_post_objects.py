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
@staticmethod
def _make_post_objects(objects):
    post_objects = []
    for o in objects:
        if isinstance(o, str):
            obj = SwiftPostObject(o)
            post_objects.append(obj)
        elif isinstance(o, SwiftPostObject):
            post_objects.append(o)
        else:
            raise SwiftError('The post operation takes only strings or SwiftPostObjects as input', obj=o)
    return post_objects