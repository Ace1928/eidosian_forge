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
@dataclass()
class Span:
    span_id: Optional[str] = field(default=None)
    name: Optional[str] = field(default=None)
    start_time_ms: Optional[int] = field(default=None)
    end_time_ms: Optional[int] = field(default=None)
    status_code: Optional[StatusCode] = field(default=None)
    status_message: Optional[str] = field(default=None)
    attributes: Optional[Dict[str, Any]] = field(default=None)
    results: Optional[List[Result]] = field(default=None)
    child_spans: Optional[List['Span']] = field(default=None)
    span_kind: Optional[SpanKind] = field(default=None)

    def add_attribute(self, key: str, value: Any) -> None:
        if self.attributes is None:
            self.attributes = {}
        self.attributes[key] = value

    def add_named_result(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        if self.results is None:
            self.results = []
        self.results.append(Result(inputs, outputs))

    def add_child_span(self, span: 'Span') -> None:
        if self.child_spans is None:
            self.child_spans = []
        self.child_spans.append(span)