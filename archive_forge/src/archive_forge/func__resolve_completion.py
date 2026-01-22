import datetime
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import wandb
from wandb.sdk.data_types import trace_tree
from wandb.sdk.integration_utils.auto_logging import Response
def _resolve_completion(self, request: Dict[str, Any], response: Response, time_elapsed: float) -> Dict[str, Any]:
    """Resolves the request and response objects for `openai.Completion`."""
    request_str = f'\n\n**Prompt**: {request['prompt']}\n'
    choices = [f'\n\n**Completion**: {choice['text']}\n' for choice in response['choices']]
    return self._resolve_metrics(request=request, response=response, request_str=request_str, choices=choices, time_elapsed=time_elapsed)