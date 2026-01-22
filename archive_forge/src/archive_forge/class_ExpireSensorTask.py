import logging
import sys
import time
import threading
from .kafka_metric import KafkaMetric
from .measurable import AnonMeasurable
from .metric_config import MetricConfig
from .metric_name import MetricName
from .stats import Sensor
class ExpireSensorTask(object):
    """
        This iterates over every Sensor and triggers a remove_sensor
        if it has expired. Package private for testing
        """

    @staticmethod
    def run(metrics):
        items = list(metrics._sensors.items())
        for name, sensor in items:
            with sensor._lock:
                if sensor.has_expired():
                    logger.debug('Removing expired sensor %s', name)
                    metrics.remove_sensor(name)