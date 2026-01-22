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
def _make_upload_objects(objects, pseudo_folder=''):
    upload_objects = []
    for o in objects:
        if isinstance(o, str):
            obj = SwiftUploadObject(o, urljoin(pseudo_folder, o.lstrip('/')))
            upload_objects.append(obj)
        elif isinstance(o, SwiftUploadObject):
            o.object_name = urljoin(pseudo_folder, o.object_name)
            upload_objects.append(o)
        else:
            raise SwiftError('The upload operation takes only strings or SwiftUploadObjects as input', obj=o)
    return upload_objects