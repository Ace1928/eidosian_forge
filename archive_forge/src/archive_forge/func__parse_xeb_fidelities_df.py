import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def _parse_xeb_fidelities_df(metrics: 'cirq_google.Calibration', super_name: str) -> pd.DataFrame:
    """Parse a fidelities DataFrame from Metric protos.

    Args:
        metrics: The metrics from a CalibrationResult
        super_name: The metric name prefix. We will extract information for metrics named like
            "{super_name}_depth_{depth}", so you can have multiple independent DataFrames in
            one CalibrationResult.
    """
    records: List[Dict[str, Union[int, float, Tuple[cirq.Qid, cirq.Qid]]]] = []
    for metric_name in metrics.keys():
        ma = re.match(f'{super_name}_depth_(\\d+)$', metric_name)
        if ma is None:
            continue
        for (layer_str, pair_str, qa, qb), (value,) in metrics[metric_name].items():
            records.append({'cycle_depth': int(ma.group(1)), 'layer_i': _get_labeled_int('layer', cast(str, layer_str)), 'pair_i': _get_labeled_int('pair', cast(str, pair_str)), 'fidelity': float(value), 'pair': (cast(cirq.GridQubit, qa), cast(cirq.GridQubit, qb))})
    return pd.DataFrame(records)