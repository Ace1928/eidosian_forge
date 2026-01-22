import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
class InfoMetricFamily(Metric):
    """A single info and its samples.

    For use by custom collectors.
    """

    def __init__(self, name: str, documentation: str, value: Optional[Dict[str, str]]=None, labels: Optional[Sequence[str]]=None):
        Metric.__init__(self, name, documentation, 'info')
        if labels is not None and value is not None:
            raise ValueError('Can only specify at most one of value and labels.')
        if labels is None:
            labels = []
        self._labelnames = tuple(labels)
        if value is not None:
            self.add_metric([], value)

    def add_metric(self, labels: Sequence[str], value: Dict[str, str], timestamp: Optional[Union[Timestamp, float]]=None) -> None:
        """Add a metric to the metric family.

        Args:
          labels: A list of label values
          value: A dict of labels
        """
        self.samples.append(Sample(self.name + '_info', dict(dict(zip(self._labelnames, labels)), **value), 1, timestamp))