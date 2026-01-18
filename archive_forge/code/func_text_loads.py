import io
import json
from itertools import islice
from typing import Any, Callable, Dict, List
import numpy as np
import pyarrow as pa
import datasets
def text_loads(data: bytes):
    return data.decode('utf-8')