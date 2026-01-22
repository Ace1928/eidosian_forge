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
def _copy_object_job(conn, container, obj, destination, headers, fresh_metadata):
    response_dict = {}
    res = {'success': True, 'action': 'copy_object', 'container': container, 'object': obj, 'destination': destination, 'headers': headers, 'fresh_metadata': fresh_metadata, 'response_dict': response_dict}
    try:
        conn.copy_object(container, obj, destination=destination, headers=headers, fresh_metadata=fresh_metadata, response_dict=response_dict)
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
    return res