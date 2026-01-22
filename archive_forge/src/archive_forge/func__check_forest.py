import threading
import time
from aiokafka.errors import QuotaViolationError
from aiokafka.metrics.kafka_metric import KafkaMetric
def _check_forest(self, sensors):
    """Validate that this sensor doesn't end up referencing itself."""
    if self in sensors:
        raise ValueError('Circular dependency in sensors: %s is its ownparent.' % (self.name,))
    sensors.add(self)
    for parent in self._parents:
        parent._check_forest(sensors)