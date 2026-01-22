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
def _submit_page_downloads(self, container, page_generator, options):
    try:
        list_page = next(page_generator)
    except StopIteration:
        return None
    if list_page['success']:
        objects = [o['name'] for o in list_page['listing']]
        if options['shuffle']:
            shuffle(objects)
        o_downs = [self.thread_manager.object_dd_pool.submit(self._download_object_job, container, obj, options) for obj in objects]
        return o_downs
    else:
        raise list_page['error']