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
def _delete_segment(conn, container, obj, results_queue=None):
    results_dict = {}
    try:
        res = {'success': True}
        conn.delete_object(container, obj, response_dict=results_dict)
    except Exception as err:
        if not isinstance(err, ClientException) or err.http_status != 404:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res = {'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time}
    res.update({'action': 'delete_segment', 'container': container, 'object': obj, 'attempts': conn.attempts, 'response_dict': results_dict})
    if results_queue is not None:
        results_queue.put(res)
    return res