import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
class NullMetricHandler(MetricHandler):

    def emit(self, metric_data: MetricData):
        pass