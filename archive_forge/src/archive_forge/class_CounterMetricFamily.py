import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
class CounterMetricFamily(Metric):
    """A single counter and its samples.

    For use by custom collectors.
    """

    def __init__(self, name: str, documentation: str, value: Optional[float]=None, labels: Optional[Sequence[str]]=None, created: Optional[float]=None, unit: str=''):
        if name.endswith('_total'):
            name = name[:-6]
        Metric.__init__(self, name, documentation, 'counter', unit)
        if labels is not None and value is not None:
            raise ValueError('Can only specify at most one of value and labels.')
        if labels is None:
            labels = []
        self._labelnames = tuple(labels)
        if value is not None:
            self.add_metric([], value, created)

    def add_metric(self, labels: Sequence[str], value: float, created: Optional[float]=None, timestamp: Optional[Union[Timestamp, float]]=None) -> None:
        """Add a metric to the metric family.

        Args:
          labels: A list of label values
          value: The value of the metric
          created: Optional unix timestamp the child was created at.
        """
        self.samples.append(Sample(self.name + '_total', dict(zip(self._labelnames, labels)), value, timestamp))
        if created is not None:
            self.samples.append(Sample(self.name + '_created', dict(zip(self._labelnames, labels)), created, timestamp))