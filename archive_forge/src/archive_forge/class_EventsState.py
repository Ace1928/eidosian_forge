import collections
import logging
import shelve
import threading
import time
from collections import Counter
from functools import partial
from celery.events import EventReceiver
from celery.events.state import State
from prometheus_client import Counter as PrometheusCounter
from prometheus_client import Gauge, Histogram
from tornado.ioloop import PeriodicCallback
from tornado.options import options
class EventsState(State):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = collections.defaultdict(Counter)
        self.metrics = get_prometheus_metrics()

    def event(self, event):
        super().event(event)
        worker_name = event['hostname']
        event_type = event['type']
        self.counter[worker_name][event_type] += 1
        if event_type.startswith('task-'):
            task_id = event['uuid']
            task = self.tasks.get(task_id)
            task_name = event.get('name', '')
            if not task_name and task_id in self.tasks:
                task_name = task.name or ''
            self.metrics.events.labels(worker_name, event_type, task_name).inc()
            runtime = event.get('runtime', 0)
            if runtime:
                self.metrics.runtime.labels(worker_name, task_name).observe(runtime)
            task_started = task.started
            task_received = task.received
            if event_type == 'task-received' and (not task.eta) and task_received:
                self.metrics.number_of_prefetched_tasks.labels(worker_name, task_name).inc()
            if event_type == 'task-started' and (not task.eta) and task_started and task_received:
                self.metrics.prefetch_time.labels(worker_name, task_name).set(task_started - task_received)
                self.metrics.number_of_prefetched_tasks.labels(worker_name, task_name).dec()
            if event_type in ['task-succeeded', 'task-failed'] and (not task.eta) and task_started and task_received:
                self.metrics.prefetch_time.labels(worker_name, task_name).set(0)
        if event_type == 'worker-online':
            self.metrics.worker_online.labels(worker_name).set(1)
        if event_type == 'worker-heartbeat':
            self.metrics.worker_online.labels(worker_name).set(1)
            num_executing_tasks = event.get('active')
            if num_executing_tasks is not None:
                self.metrics.worker_number_of_currently_executing_tasks.labels(worker_name).set(num_executing_tasks)
        if event_type == 'worker-offline':
            self.metrics.worker_online.labels(worker_name).set(0)