import datetime
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import wandb
from wandb.sdk.data_types import trace_tree
from wandb.sdk.integration_utils.auto_logging import Response
def _resolve_metrics(self, request: Dict[str, Any], response: Response, request_str: str, choices: List[str], time_elapsed: float) -> Dict[str, Any]:
    """Resolves the request and response objects for `openai.Completion`."""
    results = [trace_tree.Result(inputs={'request': request_str}, outputs={'response': choice}) for choice in choices]
    metrics = self._get_metrics_to_log(request, response, results, time_elapsed)
    return self._convert_metrics_to_dict(metrics)