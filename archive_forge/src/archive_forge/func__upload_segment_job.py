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
def _upload_segment_job(conn, path, container, segment_name, segment_start, segment_size, segment_index, obj_name, options, results_queue=None):
    results_dict = {}
    if options['segment_container']:
        segment_container = options['segment_container']
    else:
        segment_container = container + '_segments'
    res = {'action': 'upload_segment', 'for_container': container, 'for_object': obj_name, 'segment_index': segment_index, 'segment_size': segment_size, 'segment_location': '/%s/%s' % (segment_container, segment_name), 'log_line': '%s segment %s' % (obj_name, segment_index)}
    fp = None
    try:
        fp = open(path, 'rb', DISK_BUFFER)
        fp.seek(segment_start)
        contents = LengthWrapper(fp, segment_size, md5=options['checksum'])
        etag = conn.put_object(segment_container, segment_name, contents, content_length=segment_size, content_type='application/swiftclient-segment', response_dict=results_dict)
        if options['checksum'] and etag and (etag != contents.get_md5sum()):
            raise SwiftError('Segment {0}: upload verification failed: md5 mismatch, local {1} != remote {2} (remote segment has not been removed)'.format(segment_index, contents.get_md5sum(), etag))
        res.update({'success': True, 'response_dict': results_dict, 'segment_etag': etag, 'attempts': conn.attempts})
        if results_queue is not None:
            results_queue.put(res)
        return res
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': results_dict, 'attempts': conn.attempts})
        if results_queue is not None:
            results_queue.put(res)
        return res
    finally:
        if fp is not None:
            fp.close()