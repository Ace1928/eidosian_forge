import datetime
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import wandb
from wandb.sdk.data_types import trace_tree
from wandb.sdk.integration_utils.auto_logging import Response
@staticmethod
def _get_usage_metrics(response: Response, time_elapsed: float) -> UsageMetrics:
    """Gets the usage stats from the response object."""
    if response.get('usage'):
        usage_stats = UsageMetrics(**response['usage'])
    else:
        usage_stats = UsageMetrics()
    usage_stats.elapsed_time = time_elapsed
    return usage_stats