import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from typing_extensions import override
from pytorch_lightning.profilers.profiler import Profiler
def _make_report_extended(self) -> Tuple[_TABLE_DATA_EXTENDED, float, float]:
    total_duration = time.monotonic() - self.start_time
    report = []
    for a, d in self.recorded_durations.items():
        d_tensor = torch.tensor(d)
        len_d = len(d)
        sum_d = torch.sum(d_tensor).item()
        percentage_d = 100.0 * sum_d / total_duration
        report.append((a, sum_d / len_d, len_d, sum_d, percentage_d))
    report.sort(key=lambda x: x[4], reverse=True)
    total_calls = sum((x[2] for x in report))
    return (report, total_calls, total_duration)