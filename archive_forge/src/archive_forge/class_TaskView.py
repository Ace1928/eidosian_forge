import copy
import logging
from functools import total_ordering
from tornado import web
from ..utils.tasks import as_dict, get_task_by_id, iter_tasks
from ..views import BaseHandler
class TaskView(BaseHandler):

    @web.authenticated
    def get(self, task_id):
        task = get_task_by_id(self.application.events, task_id)
        if task is None:
            raise web.HTTPError(404, f"Unknown task '{task_id}'")
        task = self.format_task(task)
        self.render('task.html', task=task)