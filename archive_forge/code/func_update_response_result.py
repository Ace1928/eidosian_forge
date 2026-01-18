import json
import logging
from collections import OrderedDict
from datetime import datetime
from celery import states
from celery.backends.base import DisabledBackend
from celery.contrib.abortable import AbortableAsyncResult
from celery.result import AsyncResult
from tornado import web
from tornado.escape import json_decode
from tornado.ioloop import IOLoop
from tornado.web import HTTPError
from ..utils import tasks
from ..utils.broker import Broker
from . import BaseApiHandler
def update_response_result(self, response, result):
    if result.state == states.FAILURE:
        response.update({'result': self.safe_result(result.result), 'traceback': result.traceback})
    else:
        response.update({'result': self.safe_result(result.result)})