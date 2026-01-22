from typing import Any, Dict, Sequence
import _pytest
import pytest
from onnx.backend.test.report.coverage import Coverage
def _add_mark(mark: Any, bucket: str) -> None:
    proto = mark.args[0]
    if isinstance(proto, list):
        assert len(proto) == 1
        proto = proto[0]
    if proto is not None:
        _coverage.add_proto(proto, bucket, mark.args[1] == 'RealModel')