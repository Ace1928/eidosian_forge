import logging
import time
from tornado import web
from ..options import options
from ..views import BaseHandler
class WorkerView(BaseHandler):

    @web.authenticated
    async def get(self, name):
        try:
            self.application.update_workers(workername=name)
        except Exception as e:
            logger.error(e)
        worker = self.application.workers.get(name)
        if worker is None:
            raise web.HTTPError(404, f"Unknown worker '{name}'")
        if 'stats' not in worker:
            raise web.HTTPError(404, f"Unable to get stats for '{name}' worker")
        self.render('worker.html', worker=dict(worker, name=name))