import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@dataclass
class _RunConfigDefaults(_RunDefaults):
    batch_sizes: Optional[List[int]] = field(default_factory=lambda: [4, 8], metadata={'description': 'Batch sizes to include in the run to measure time metrics.'})
    input_lengths: Optional[List[int]] = field(default_factory=lambda: [128], metadata={'description': 'Input lengths to include in the run to measure time metrics.'})