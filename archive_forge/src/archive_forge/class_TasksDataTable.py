import copy
import logging
from functools import total_ordering
from tornado import web
from ..utils.tasks import as_dict, get_task_by_id, iter_tasks
from ..views import BaseHandler
class TasksDataTable(BaseHandler):

    @web.authenticated
    def get(self):
        app = self.application
        draw = self.get_argument('draw', type=int)
        start = self.get_argument('start', type=int)
        length = self.get_argument('length', type=int)
        search = self.get_argument('search[value]', type=str)
        column = self.get_argument('order[0][column]', type=int)
        sort_by = self.get_argument(f'columns[{column}][data]', type=str)
        sort_order = self.get_argument('order[0][dir]', type=str) == 'desc'

        def key(item):
            return Comparable(getattr(item[1], sort_by))
        self.maybe_normalize_for_sort(app.events.state.tasks_by_timestamp(), sort_by)
        sorted_tasks = sorted(iter_tasks(app.events, search=search), key=key, reverse=sort_order)
        filtered_tasks = []
        for task in sorted_tasks[start:start + length]:
            task_dict = as_dict(self.format_task(task)[1])
            if task_dict.get('worker'):
                task_dict['worker'] = task_dict['worker'].hostname
            filtered_tasks.append(task_dict)
        self.write(dict(draw=draw, data=filtered_tasks, recordsTotal=len(sorted_tasks), recordsFiltered=len(sorted_tasks)))

    @classmethod
    def maybe_normalize_for_sort(cls, tasks, sort_by):
        sort_keys = {'name': str, 'state': str, 'received': float, 'started': float, 'runtime': float}
        if sort_by in sort_keys:
            for _, task in tasks:
                attr_value = getattr(task, sort_by, None)
                if attr_value:
                    try:
                        setattr(task, sort_by, sort_keys[sort_by](attr_value))
                    except TypeError:
                        pass

    @web.authenticated
    def post(self):
        return self.get()

    def format_task(self, task):
        uuid, args = task
        custom_format_task = self.application.options.format_task
        if custom_format_task:
            try:
                args = custom_format_task(copy.copy(args))
            except Exception:
                logger.exception("Failed to format '%s' task", uuid)
        return (uuid, args)