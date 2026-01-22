import collections
import os
import time
from dataclasses import dataclass
from typing import (
import numpy as np
import ray
from ray import DynamicObjectRefGenerator
from ray.data._internal.util import _check_pyarrow_version, _truncated_repr
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI
import psutil
class _CallableClassProtocol(Protocol[T, U]):

    def __call__(self, __arg: T) -> Union[U, Iterator[U]]:
        ...