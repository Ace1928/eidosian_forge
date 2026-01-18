from typing import Any, Dict, Sequence
from wandb.sdk.integration_utils.auto_logging import Response
from .resolvers import (
Main call method for the `DiffusersPipelineResolver` class.

        Arguments:
            args: (Sequence[Any]) List of arguments.
            kwargs: (Dict[str, Any]) Dictionary of keyword arguments.
            response: (wandb.sdk.integration_utils.auto_logging.Response) The response from
                the request.
            start_time: (float) Time when request started.
            time_elapsed: (float) Time elapsed for the request.

        Returns:
            Packed data as a dictionary for logging to wandb, None if an exception occurred.
        