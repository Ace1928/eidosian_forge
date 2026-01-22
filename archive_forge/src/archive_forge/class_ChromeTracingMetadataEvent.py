import os
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Union
import ray
@dataclass(init=True)
class ChromeTracingMetadataEvent:
    name: str
    args: Dict[str, str]
    pid: int
    tid: int = None
    ph: str = 'M'