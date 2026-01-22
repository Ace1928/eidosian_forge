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
class GetQueueLengths(BaseTaskHandler):

    @web.authenticated
    async def get(self):
        """
Return length of all active queues

**Example request**:

.. sourcecode:: http

  GET /api/queues/length
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 94
  Content-Type: application/json; charset=UTF-8

  {
      "active_queues": [
          {"name": "celery", "messages": 0},
          {"name": "video-queue", "messages": 5}
      ]
  }

:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 503: result backend is not configured
        """
        app = self.application
        http_api = None
        if app.transport == 'amqp' and app.options.broker_api:
            http_api = app.options.broker_api
        broker = Broker(app.capp.connection().as_uri(include_password=True), http_api=http_api, broker_options=self.capp.conf.broker_transport_options, broker_use_ssl=self.capp.conf.broker_use_ssl)
        queues = await broker.queues(self.get_active_queue_names())
        self.write({'active_queues': queues})