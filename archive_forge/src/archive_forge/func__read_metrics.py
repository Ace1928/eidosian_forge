from collections import defaultdict
import glob
import json
import os
import warnings
from .metrics import Gauge
from .metrics_core import Metric
from .mmap_dict import MmapedDict
from .samples import Sample
from .utils import floatToGoString
@staticmethod
def _read_metrics(files):
    metrics = {}
    key_cache = {}

    def _parse_key(key):
        val = key_cache.get(key)
        if not val:
            metric_name, name, labels, help_text = json.loads(key)
            labels_key = tuple(sorted(labels.items()))
            val = key_cache[key] = (metric_name, name, labels, labels_key, help_text)
        return val
    for f in files:
        parts = os.path.basename(f).split('_')
        typ = parts[0]
        try:
            file_values = MmapedDict.read_all_values_from_file(f)
        except FileNotFoundError:
            if typ == 'gauge' and parts[1].startswith('live'):
                continue
            raise
        for key, value, timestamp, _ in file_values:
            metric_name, name, labels, labels_key, help_text = _parse_key(key)
            metric = metrics.get(metric_name)
            if metric is None:
                metric = Metric(metric_name, help_text, typ)
                metrics[metric_name] = metric
            if typ == 'gauge':
                pid = parts[2][:-3]
                metric._multiprocess_mode = parts[1]
                metric.add_sample(name, labels_key + (('pid', pid),), value, timestamp)
            else:
                metric.add_sample(name, labels_key, value)
    return metrics