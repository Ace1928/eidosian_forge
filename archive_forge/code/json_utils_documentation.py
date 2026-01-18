import collections
import enum
import json
import numpy as np
import wrapt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
Encodes objects for types that aren't handled by the default encoder.