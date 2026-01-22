import os
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Union
import ray
class _NullLogSpan:
    """A log span context manager that does nothing"""

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass