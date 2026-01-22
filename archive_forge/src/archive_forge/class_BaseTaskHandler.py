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
class BaseTaskHandler(BaseApiHandler):
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

    def get_task_args(self):
        try:
            body = self.request.body
            options = json_decode(body) if body else {}
        except ValueError as e:
            raise HTTPError(400, str(e)) from e
        if not isinstance(options, dict):
            raise HTTPError(400, 'invalid options')
        args = options.pop('args', [])
        kwargs = options.pop('kwargs', {})
        if not isinstance(args, (list, tuple)):
            raise HTTPError(400, 'args must be an array')
        return (args, kwargs, options)

    @staticmethod
    def backend_configured(result):
        return not isinstance(result.backend, DisabledBackend)

    def write_error(self, status_code, **kwargs):
        self.set_status(status_code)

    def update_response_result(self, response, result):
        if result.state == states.FAILURE:
            response.update({'result': self.safe_result(result.result), 'traceback': result.traceback})
        else:
            response.update({'result': self.safe_result(result.result)})

    def normalize_options(self, options):
        if 'eta' in options:
            options['eta'] = datetime.strptime(options['eta'], self.DATE_FORMAT)
        if 'countdown' in options:
            options['countdown'] = float(options['countdown'])
        if 'expires' in options:
            expires = options['expires']
            try:
                expires = float(expires)
            except ValueError:
                expires = datetime.strptime(expires, self.DATE_FORMAT)
            options['expires'] = expires

    def safe_result(self, result):
        """returns json encodable result"""
        try:
            json.dumps(result)
        except TypeError:
            return repr(result)
        return result