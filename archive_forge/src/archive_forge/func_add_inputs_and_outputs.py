import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.data_types
from wandb.sdk.data_types import _dtypes
from wandb.sdk.data_types.base_types.media import Media
def add_inputs_and_outputs(self, inputs: dict, outputs: dict) -> 'Trace':
    """Add a result to the span of the current trace.

        Args:
            inputs: Dictionary of inputs to be logged with the span.
            outputs: Dictionary of outputs to be logged with the span.

        Returns:
            The current trace object with the result added to it.
        """
    if self._span.results is None:
        result = Result(inputs=inputs, outputs=outputs)
        self._span.results = [result]
    else:
        result = Result(inputs=inputs, outputs=outputs)
        self._span.results.append(result)
    return self