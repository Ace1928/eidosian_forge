import abc
import base64
import collections
import pickle
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from ray.air.util.data_batch_conversion import BatchFormat
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@Deprecated
def transform_stats(self) -> Optional[str]:
    """Return Dataset stats for the most recent transform call, if any."""
    raise DeprecationWarning('`preprocessor.transform_stats()` is no longer supported in Ray 2.4. With Dataset now lazy by default, the stats are only populated after execution. Once the dataset transform is executed, the stats can be accessed directly from the transformed dataset (`ds.stats()`), or can be viewed in the ray-data.log file saved in the Ray logs directory (defaults to /tmp/ray/session_{SESSION_ID}/logs/).')