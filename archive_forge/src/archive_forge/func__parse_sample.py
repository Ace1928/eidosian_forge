import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def _parse_sample(text):
    separator = ' # '
    label_start = text.find('{')
    if label_start == -1 or separator in text[:label_start]:
        name_end = text.index(' ')
        name = text[:name_end]
        remaining_text = text[name_end + 1:]
        value, timestamp, exemplar = _parse_remaining_text(remaining_text)
        return Sample(name, {}, value, timestamp, exemplar)
    name = text[:label_start]
    if separator not in text:
        label_end = text.rindex('}')
        label = text[label_start + 1:label_end]
        labels = _parse_labels(label)
    else:
        labels, labels_len = _parse_labels_with_state_machine(text[label_start + 1:])
        label_end = labels_len + len(name)
    remaining_text = text[label_end + 2:]
    value, timestamp, exemplar = _parse_remaining_text(remaining_text)
    return Sample(name, labels, value, timestamp, exemplar)