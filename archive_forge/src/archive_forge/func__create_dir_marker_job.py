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
def _create_dir_marker_job(conn, container, obj, options, path=None):
    res = {'action': 'create_dir_marker', 'container': container, 'object': obj, 'path': path}
    results_dict = {}
    if obj.startswith('./') or obj.startswith('.\\'):
        obj = obj[2:]
    if obj.startswith('/'):
        obj = obj[1:]
    if path is not None:
        put_headers = {'x-object-meta-mtime': '%f' % getmtime(path)}
    else:
        put_headers = {'x-object-meta-mtime': '%f' % round(time())}
    res['headers'] = put_headers
    if options['changed']:
        try:
            headers = conn.head_object(container, obj)
            ct = headers.get('content-type', '').split(';', 1)[0]
            cl = int(headers.get('content-length'))
            et = headers.get('etag')
            mt = headers.get('x-object-meta-mtime')
            if ct in KNOWN_DIR_MARKERS and cl == 0 and (et == EMPTY_ETAG) and (mt == put_headers['x-object-meta-mtime']):
                res['success'] = True
                return res
        except ClientException as err:
            if err.http_status != 404:
                traceback, err_time = report_traceback()
                logger.exception(err)
                res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                return res
    try:
        conn.put_object(container, obj, '', content_length=0, content_type=KNOWN_DIR_MARKERS[0], headers=put_headers, response_dict=results_dict)
        res.update({'success': True, 'response_dict': results_dict})
        return res
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': results_dict})
        return res