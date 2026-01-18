from typing import Any, List
import ray
from ray import cloudpickle
Efficiently estimates the Ray serialized size of a stream of items.

    For efficiency, this only samples a fraction of the added items for real
    Ray-serialization.
    