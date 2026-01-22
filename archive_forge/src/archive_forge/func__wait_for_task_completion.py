import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _wait_for_task_completion(self, task_href, timeout=DEFAULT_TASK_COMPLETION_TIMEOUT):
    start_time = time.time()
    res = self.connection.request(get_url_path(task_href))
    status = res.object.get('status')
    while status != 'success':
        if status == 'error':
            error_elem = res.object.find(fixxpath(res.object, 'Error'))
            error_msg = 'Unknown error'
            if error_elem is not None:
                error_msg = error_elem.get('message')
            raise Exception('Error status returned by task {}.: {}'.format(task_href, error_msg))
        if status == 'canceled':
            raise Exception('Canceled status returned by task %s.' % task_href)
        if time.time() - start_time >= timeout:
            raise Exception('Timeout ({} sec) while waiting for task {}.'.format(timeout, task_href))
        time.sleep(5)
        res = self.connection.request(get_url_path(task_href))
        status = res.object.get('status')