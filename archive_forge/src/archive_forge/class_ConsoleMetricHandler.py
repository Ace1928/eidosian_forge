import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
class ConsoleMetricHandler(MetricHandler):

    def emit(self, metric_data: MetricData):
        print(f'[{metric_data.timestamp}][{metric_data.group_name}]: {metric_data.name}={metric_data.value}')