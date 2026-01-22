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
class TaskAsyncApply(BaseTaskHandler):

    @web.authenticated
    def post(self, taskname):
        """
Execute a task

**Example request**:

.. sourcecode:: http

  POST /api/task/async-apply/tasks.add HTTP/1.1
  Accept: application/json
  Accept-Encoding: gzip, deflate, compress
  Content-Length: 16
  Content-Type: application/json; charset=utf-8
  Host: localhost:5555

  {
      "args": [1, 2]
  }

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 71
  Content-Type: application/json; charset=UTF-8
  Date: Sun, 13 Apr 2014 15:55:00 GMT

  {
      "state": "PENDING",
      "task-id": "abc300c7-2922-4069-97b6-a635cc2ac47c"
  }

:query args: a list of arguments
:query kwargs: a dictionary of arguments
:query options: a dictionary of `apply_async` keyword arguments
:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 404: unknown task
        """
        args, kwargs, options = self.get_task_args()
        logger.debug("Invoking a task '%s' with '%s' and '%s'", taskname, args, kwargs)
        try:
            task = self.capp.tasks[taskname]
        except KeyError as exc:
            raise HTTPError(404, f"Unknown task '{taskname}'") from exc
        try:
            self.normalize_options(options)
        except ValueError as exc:
            raise HTTPError(400, 'Invalid option') from exc
        result = task.apply_async(args=args, kwargs=kwargs, **options)
        response = {'task-id': result.task_id}
        if self.backend_configured(result):
            response.update(state=result.state)
        self.write(response)