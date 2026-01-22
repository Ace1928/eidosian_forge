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
def add_child(self, child: 'Trace') -> 'Trace':
    """Utility to add a child span to the current span of the trace.

        Args:
            child: The child span to be added to the current span of the trace.

        Returns:
            The current trace object with the child span added to it.
        """
    self._span.add_child_span(child._span)
    if self._model_dict is not None and child._model_dict is not None:
        self._model_dict.update({child._span.name: child._model_dict})
    return self