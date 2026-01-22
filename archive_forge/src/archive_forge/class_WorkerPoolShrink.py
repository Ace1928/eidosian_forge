import logging
from tornado import web
from . import BaseApiHandler
class WorkerPoolShrink(ControlHandler):

    @web.authenticated
    def post(self, workername):
        """
Shrink worker's pool

**Example request**:

.. sourcecode:: http

  POST /api/worker/pool/shrink/celery@worker2 HTTP/1.1
  Content-Length: 0
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 60
  Content-Type: application/json; charset=UTF-8

  {
      "message": "Shrinking 'celery@worker2' worker's pool by 1"
  }

:query n: number of pool processes to shrink, default is 1
:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 403: failed to shrink
:statuscode 404: unknown worker
        """
        if not self.is_worker(workername):
            raise web.HTTPError(404, f"Unknown worker '{workername}'")
        n = self.get_argument('n', default=1, type=int)
        logger.info("Shrinking '%s' worker's pool by '%s'", workername, n)
        response = self.capp.control.pool_shrink(n=n, reply=True, destination=[workername])
        if response and 'ok' in response[0][workername]:
            self.write(dict(message=f"Shrinking '{workername}' worker's pool by {n}"))
        else:
            logger.error(response)
            self.set_status(403)
            reason = self.error_reason(workername, response)
            self.write(f"Failed to shrink '{workername}' worker's pool: {reason}")