import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
class StateSetMetricFamily(Metric):
    """A single stateset and its samples.

    For use by custom collectors.
    """

    def __init__(self, name: str, documentation: str, value: Optional[Dict[str, bool]]=None, labels: Optional[Sequence[str]]=None):
        Metric.__init__(self, name, documentation, 'stateset')
        if labels is not None and value is not None:
            raise ValueError('Can only specify at most one of value and labels.')
        if labels is None:
            labels = []
        self._labelnames = tuple(labels)
        if value is not None:
            self.add_metric([], value)

    def add_metric(self, labels: Sequence[str], value: Dict[str, bool], timestamp: Optional[Union[Timestamp, float]]=None) -> None:
        """Add a metric to the metric family.

        Args:
          labels: A list of label values
          value: A dict of string state names to booleans
        """
        labels = tuple(labels)
        for state, enabled in sorted(value.items()):
            v = 1 if enabled else 0
            self.samples.append(Sample(self.name, dict(zip(self._labelnames + (self.name,), labels + (state,))), v, timestamp))