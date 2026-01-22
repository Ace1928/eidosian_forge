import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from ray._raylet import (
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class Gauge(Metric):
    """Gauges keep the last recorded value and drop everything before.

    Unlike counters, gauges can go up or down over time.

    This corresponds to Prometheus' gauge metric:
    https://prometheus.io/docs/concepts/metric_types/#gauge

    Args:
        name: Name of the metric.
        description: Description of the metric.
        tag_keys: Tag keys of the metric.
    """

    def __init__(self, name: str, description: str='', tag_keys: Optional[Tuple[str, ...]]=None):
        super().__init__(name, description, tag_keys)
        self._metric = CythonGauge(self._name, self._description, self._tag_keys)

    def set(self, value: Union[int, float], tags: Dict[str, str]=None):
        """Set the gauge to the given `value`.

        Tags passed in will take precedence over the metric's default tags.

        Args:
            value(int, float): Value to set the gauge to.
            tags(Dict[str, str]): Tags to set or override for this gauge.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f'value must be int or float, got {type(value)}.')
        self.record(value, tags, _internal=True)

    def __reduce__(self):
        deserializer = Gauge
        serialized_data = (self._name, self._description, self._tag_keys)
        return (deserializer, serialized_data)