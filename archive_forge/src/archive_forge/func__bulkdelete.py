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
def _bulkdelete(conn, container, objects, options):
    results_dict = {}
    try:
        headers = {'Accept': 'application/json', 'Content-Type': 'text/plain'}
        res = {'container': container, 'objects': objects}
        objects = [quote(('/%s/%s' % (container, obj)).encode('utf-8')) for obj in objects]
        headers, body = conn.post_account(headers=headers, query_string='bulk-delete', data=b''.join((obj.encode('utf-8') + b'\n' for obj in objects)), response_dict=results_dict)
        if body:
            res.update({'success': True, 'result': parse_api_response(headers, body)})
        else:
            res.update({'success': False, 'error': SwiftError('No content received on account POST. Is the bulk operations middleware enabled?')})
    except Exception as e:
        traceback, err_time = report_traceback()
        logger.exception(e)
        res.update({'success': False, 'error': e, 'traceback': traceback})
    res.update({'action': 'bulk_delete', 'attempts': conn.attempts, 'response_dict': results_dict})
    return res