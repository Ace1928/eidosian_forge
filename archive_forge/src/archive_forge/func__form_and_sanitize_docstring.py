import argparse
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Optional
import pydot
from onnx import GraphProto, ModelProto, NodeProto
def _form_and_sanitize_docstring(s: str) -> str:
    url = 'javascript:alert('
    url += _escape_label(s).replace('"', "'").replace('<', '').replace('>', '')
    url += ')'
    return url