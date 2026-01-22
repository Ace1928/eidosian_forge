import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def _group_for_sample(sample, name, typ):
    if typ == 'info':
        return {}
    if typ == 'summary' and sample.name == name:
        d = sample.labels.copy()
        del d['quantile']
        return d
    if typ == 'stateset':
        d = sample.labels.copy()
        del d[name]
        return d
    if typ in ['histogram', 'gaugehistogram'] and sample.name == name + '_bucket':
        d = sample.labels.copy()
        del d['le']
        return d
    return sample.labels