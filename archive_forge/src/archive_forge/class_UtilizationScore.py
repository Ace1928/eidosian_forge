import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.core.generated.common_pb2 import PlacementStrategy
class UtilizationScore:
    """This fancy class just defines the `UtilizationScore` protocol to be
    some type that is a "totally ordered set" (i.e. things that can be sorted).

    What we're really trying to express is

    ```
    UtilizationScore = TypeVar("UtilizationScore", bound=Comparable["UtilizationScore"])
    ```

    but Comparable isn't a real type and, and a bound with a type argument
    can't be enforced (f-bounded polymorphism with contravariance). See Guido's
    comment for more details: https://github.com/python/typing/issues/59.

    This isn't just a `float`. In the case of the default scorer, it's a
    `Tuple[float, float]` which is quite difficult to map to a single number.

    """

    @abstractmethod
    def __eq__(self, other: 'UtilizationScore') -> bool:
        pass

    @abstractmethod
    def __lt__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        pass

    def __gt__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        return not self < other and self != other

    def __le__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        return self < other or self == other

    def __ge__(self: 'UtilizationScore', other: 'UtilizationScore') -> bool:
        return not self < other