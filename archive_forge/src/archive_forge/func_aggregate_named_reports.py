import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
def aggregate_named_reports(named_reports: Dict[str, Dict[str, Metric]], micro_average: bool=False) -> Dict[str, Metric]:
    """
    Aggregate metrics from multiple reports.

    :param reports:
        Dict of tasks -> metrics.
    :param micro_average:
        If true, top level metrics will be the micro average. By default, we
        use macro average.
    :return:
        The aggregated report
    """
    if len(named_reports) == 0:
        raise ValueError('Cannot aggregate empty reports.')
    if len(named_reports) == 1:
        return next(iter(named_reports.values()))
    m: Dict[str, Metric] = {}
    macro_averages: Dict[str, Dict[str, Metric]] = {}
    for task_id, task_report in named_reports.items():
        for each_metric, value in task_report.items():
            if value.is_global:
                if each_metric not in m:
                    m[each_metric] = value
            else:
                task_metric = f'{task_id}/{each_metric}'
                m[task_metric] = m.get(task_metric) + value
                if micro_average or not value.macro_average:
                    m[each_metric] = m.get(each_metric) + value
                else:
                    if each_metric not in macro_averages:
                        macro_averages[each_metric] = {}
                    macro_averages[each_metric][task_id] = value
    for key, values in macro_averages.items():
        m[key] = MacroAverageMetric(values)
    return m