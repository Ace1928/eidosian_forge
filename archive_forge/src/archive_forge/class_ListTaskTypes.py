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
class ListTaskTypes(BaseTaskHandler):

    @web.authenticated
    def get(self):
        """
List (seen) task types

**Example request**:

.. sourcecode:: http

  GET /api/task/types HTTP/1.1
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 44
  Content-Type: application/json; charset=UTF-8

  {
      "task-types": [
          "tasks.add",
          "tasks.sleep"
      ]
  }

:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
        """
        seen_task_types = self.application.events.state.task_types()
        response = {}
        response['task-types'] = seen_task_types
        self.write(response)