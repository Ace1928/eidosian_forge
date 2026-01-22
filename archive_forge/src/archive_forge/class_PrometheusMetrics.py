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
class PrometheusMetrics:

    def __init__(self):
        self.events = PrometheusCounter('flower_events_total', 'Number of events', ['worker', 'type', 'task'])
        self.runtime = Histogram('flower_task_runtime_seconds', 'Task runtime', ['worker', 'task'], buckets=options.task_runtime_metric_buckets)
        self.prefetch_time = Gauge('flower_task_prefetch_time_seconds', 'The time the task spent waiting at the celery worker to be executed.', ['worker', 'task'])
        self.number_of_prefetched_tasks = Gauge('flower_worker_prefetched_tasks', 'Number of tasks of given type prefetched at a worker', ['worker', 'task'])
        self.worker_online = Gauge('flower_worker_online', 'Worker online status', ['worker'])
        self.worker_number_of_currently_executing_tasks = Gauge('flower_worker_number_of_currently_executing_tasks', 'Number of tasks currently executing at a worker', ['worker'])