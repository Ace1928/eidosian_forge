import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import kubernetes_asyncio  # type: ignore # noqa: F401
import urllib3
from kubernetes_asyncio import watch
from kubernetes_asyncio.client import (  # type: ignore  # noqa: F401
import wandb
from wandb.sdk.launch.agent import LaunchAgent
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.runner.abstract import State, Status
from wandb.sdk.launch.utils import get_kube_context_and_api_client
def _state_from_conditions(conditions: List[Dict[str, Any]]) -> Optional[State]:
    """Get the status from the pod conditions."""
    true_conditions = [c.get('type', '').lower() for c in conditions if c.get('status') == 'True']
    detected_states = {CRD_STATE_DICT[c] for c in true_conditions if c in CRD_STATE_DICT}
    states_in_order: List[State] = ['finished', 'failed', 'stopping', 'running', 'starting']
    for state in states_in_order:
        if state in detected_states:
            return state
    return None