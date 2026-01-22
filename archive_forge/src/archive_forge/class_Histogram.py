import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from ray._raylet import (
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class Histogram(Metric):
    """Tracks the size and number of events in buckets.

    Histograms allow you to calculate aggregate quantiles
    such as 25, 50, 95, 99 percentile latency for an RPC.

    This corresponds to Prometheus' histogram metric:
    https://prometheus.io/docs/concepts/metric_types/#histogram

    Args:
        name: Name of the metric.
        description: Description of the metric.
        boundaries: Boundaries of histogram buckets.
        tag_keys: Tag keys of the metric.
    """

    def __init__(self, name: str, description: str='', boundaries: List[float]=None, tag_keys: Optional[Tuple[str, ...]]=None):
        super().__init__(name, description, tag_keys)
        if boundaries is None or len(boundaries) == 0:
            raise ValueError('boundaries argument should be provided when using the Histogram class. e.g., Histogram("name", boundaries=[1.0, 2.0])')
        for i, boundary in enumerate(boundaries):
            if boundary <= 0:
                raise ValueError(f'Invalid `boundaries` argument at index {i}, {boundaries}. Use positive values for the arguments.')
        self.boundaries = boundaries
        self._metric = CythonHistogram(self._name, self._description, self.boundaries, self._tag_keys)

    def observe(self, value: Union[int, float], tags: Dict[str, str]=None):
        """Observe a given `value` and add it to the appropriate bucket.

        Tags passed in will take precedence over the metric's default tags.

        Args:
            value(int, float): Value to set the gauge to.
            tags(Dict[str, str]): Tags to set or override for this gauge.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f'value must be int or float, got {type(value)}.')
        self.record(value, tags, _internal=True)

    def __reduce__(self):
        deserializer = Histogram
        serialized_data = (self._name, self._description, self.boundaries, self._tag_keys)
        return (deserializer, serialized_data)

    @property
    def info(self):
        """Return information about histogram metric."""
        info = super().info
        info.update({'boundaries': self.boundaries})
        return info