import abc
import base64
import collections
import pickle
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from ray.air.util.data_batch_conversion import BatchFormat
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
Load the original preprocessor serialized via `self.serialize()`.