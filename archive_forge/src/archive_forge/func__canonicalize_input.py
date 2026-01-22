import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
def _canonicalize_input(self, instances, signature):
    """Preprocess single-input instances to be dicts if they aren't already."""
    if not self.is_single_input(signature):
        return instances
    tensor_name = list(signature.inputs.keys())[0]
    return canonicalize_single_tensor_input(instances, tensor_name)