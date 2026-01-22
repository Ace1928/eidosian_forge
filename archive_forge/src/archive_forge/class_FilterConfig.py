import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
@dataclass
class FilterConfig:
    deduplicate_md5: bool
    piece_split_delay: float
    min_piece_length: float

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)