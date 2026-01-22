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
def _accumulate_metrics(metrics, accumulate):
    for metric in metrics.values():
        samples = defaultdict(float)
        sample_timestamps = defaultdict(float)
        buckets = defaultdict(lambda: defaultdict(float))
        samples_setdefault = samples.setdefault
        for s in metric.samples:
            name, labels, value, timestamp, exemplar = s
            if metric.type == 'gauge':
                without_pid_key = (name, tuple((l for l in labels if l[0] != 'pid')))
                if metric._multiprocess_mode in ('min', 'livemin'):
                    current = samples_setdefault(without_pid_key, value)
                    if value < current:
                        samples[without_pid_key] = value
                elif metric._multiprocess_mode in ('max', 'livemax'):
                    current = samples_setdefault(without_pid_key, value)
                    if value > current:
                        samples[without_pid_key] = value
                elif metric._multiprocess_mode in ('sum', 'livesum'):
                    samples[without_pid_key] += value
                elif metric._multiprocess_mode in ('mostrecent', 'livemostrecent'):
                    current_timestamp = sample_timestamps[without_pid_key]
                    timestamp = float(timestamp or 0)
                    if current_timestamp < timestamp:
                        samples[without_pid_key] = value
                        sample_timestamps[without_pid_key] = timestamp
                else:
                    samples[name, labels] = value
            elif metric.type == 'histogram':
                for l in labels:
                    if l[0] == 'le':
                        bucket_value = float(l[1])
                        without_le = tuple((l for l in labels if l[0] != 'le'))
                        buckets[without_le][bucket_value] += value
                        break
                else:
                    samples[name, labels] += value
            else:
                samples[name, labels] += value
        if metric.type == 'histogram':
            for labels, values in buckets.items():
                acc = 0.0
                for bucket, value in sorted(values.items()):
                    sample_key = (metric.name + '_bucket', labels + (('le', floatToGoString(bucket)),))
                    if accumulate:
                        acc += value
                        samples[sample_key] = acc
                    else:
                        samples[sample_key] = value
                if accumulate:
                    samples[metric.name + '_count', labels] = acc
        metric.samples = [Sample(name_, dict(labels), value) for (name_, labels), value in samples.items()]
    return metrics.values()