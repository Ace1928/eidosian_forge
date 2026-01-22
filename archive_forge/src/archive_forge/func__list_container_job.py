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
def _list_container_job(conn, container, options, result_queue):
    marker = options.get('marker', '')
    version_marker = options.get('version_marker', '')
    error = None
    req_headers = split_headers(options.get('header', []))
    if options.get('versions', False):
        query_string = 'versions=true'
    else:
        query_string = None
    try:
        while True:
            _, items = conn.get_container(container, marker=marker, version_marker=version_marker, prefix=options['prefix'], delimiter=options['delimiter'], headers=req_headers, query_string=query_string)
            if not items:
                result_queue.put(None)
                return
            res = {'action': 'list_container_part', 'container': container, 'prefix': options['prefix'], 'success': True, 'marker': marker, 'listing': items}
            result_queue.put(res)
            marker = items[-1].get('name', items[-1].get('subdir'))
            version_marker = items[-1].get('version_id', '')
    except ClientException as err:
        traceback, err_time = report_traceback()
        if err.http_status != 404:
            logger.exception(err)
            error = (err, traceback, err_time)
        else:
            error = (SwiftError('Container %r not found' % container, container=container, exc=err), traceback, err_time)
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        error = (err, traceback, err_time)
    res = {'action': 'list_container_part', 'container': container, 'prefix': options['prefix'], 'success': False, 'marker': marker, 'version_marker': version_marker, 'error': error[0], 'traceback': error[1], 'error_timestamp': error[2]}
    result_queue.put(res)
    result_queue.put(None)