from vllm.logger import init_logger
from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, disable_created_metrics
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
def _local_interval_elapsed(self, now: float) -> bool:
    elapsed_time = now - self.last_local_log
    return elapsed_time > self.local_interval