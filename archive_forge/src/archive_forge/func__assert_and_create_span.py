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
def _assert_and_create_span(self, name: str, kind: Optional[str]=None, status_code: Optional[str]=None, status_message: Optional[str]=None, metadata: Optional[dict]=None, start_time_ms: Optional[int]=None, end_time_ms: Optional[int]=None, inputs: Optional[dict]=None, outputs: Optional[dict]=None) -> Span:
    """Utility to assert the validity of the span parameters and create a span object.

        Args:
            name: The name of the span.
            kind: The kind of the span.
            status_code: The status code of the span.
            status_message: The status message of the span.
            metadata: Dictionary of metadata to be logged with the span.
            start_time_ms: Start time of the span in milliseconds.
            end_time_ms: End time of the span in milliseconds.
            inputs: Dictionary of inputs to be logged with the span.
            outputs: Dictionary of outputs to be logged with the span.

        Returns:
            A Span object.
        """
    if kind is not None:
        assert kind.upper() in SpanKind.__members__, "Invalid span kind, can be one of 'LLM', 'AGENT', 'CHAIN', 'TOOL'"
        kind = SpanKind(kind.upper())
    if status_code is not None:
        assert status_code.upper() in StatusCode.__members__, "Invalid status code, can be one of 'SUCCESS' or 'ERROR'"
        status_code = StatusCode(status_code.upper())
    if inputs is not None:
        assert isinstance(inputs, dict), 'Inputs must be a dictionary'
    if outputs is not None:
        assert isinstance(outputs, dict), 'Outputs must be a dictionary'
    if inputs or outputs:
        result = Result(inputs=inputs, outputs=outputs)
    else:
        result = None
    if metadata is not None:
        assert isinstance(metadata, dict), 'Metadata must be a dictionary'
    return Span(name=name, span_kind=kind, status_code=status_code, status_message=status_message, attributes=metadata, start_time_ms=start_time_ms, end_time_ms=end_time_ms, results=[result] if result else None)