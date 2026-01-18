import logging
from typing import Any, Dict, List, Sequence
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from .utils import (
Prepare the loggable dictionary, which is the packed data as a dictionary for logging to wandb, None if an exception occurred.

        Arguments:
            pipeline: (Any) The Diffusion Pipeline.
            response: (wandb.sdk.integration_utils.auto_logging.Response) The response from
                the request.
            kwargs: (Dict[str, Any]) Dictionary of keyword arguments.

        Returns:
            Packed data as a dictionary for logging to wandb, None if an exception occurred.
        