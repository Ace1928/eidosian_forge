import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
class Frameworks(str, Enum):
    onnxruntime = 'onnxruntime'