import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pytz
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def _get_model(self, pipe) -> Optional[Any]:
    """Extracts model from the pipeline.

        :param pipe: the HuggingFace pipeline
        :returns: Model if available, None otherwise
        """
    model = pipe.model
    try:
        return model.model
    except AttributeError:
        logger.info('Model does not have a `.model` attribute. Assuming `pipe.model` is the correct model.')
        return model