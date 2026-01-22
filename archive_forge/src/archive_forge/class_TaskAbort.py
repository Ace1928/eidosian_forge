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
class TaskAbort(BaseTaskHandler):

    @web.authenticated
    def post(self, taskid):
        """
Abort a running task

**Example request**:

.. sourcecode:: http

  POST /api/task/abort/c60be250-fe52-48df-befb-ac66174076e6 HTTP/1.1
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 61
  Content-Type: application/json; charset=UTF-8

  {
      "message": "Aborted '1480b55c-b8b2-462c-985e-24af3e9158f9'"
  }

:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 503: result backend is not configured
        """
        logger.info("Aborting task '%s'", taskid)
        result = AbortableAsyncResult(taskid)
        if not self.backend_configured(result):
            raise HTTPError(503)
        result.abort()
        self.write(dict(message=f"Aborted '{taskid}'"))