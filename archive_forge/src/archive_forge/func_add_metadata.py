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
def add_metadata(self, metadata: dict) -> 'Trace':
    """Add metadata to the span of the current trace."""
    if self._span.attributes is None:
        self._span.attributes = metadata
    else:
        self._span.attributes.update(metadata)
    return self