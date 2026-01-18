import base64
import collections
import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import timeit
from ._interfaces import Model
import six
from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import
Gets model signature of inputs and outputs.

    Currently only used for Tensorflow model. May be extended for use with
    XGBoost and Sklearn in the future.

    Args:
      signature_name: str of name of signature

    Returns:
      (str, SignatureDef): signature key, SignatureDef
    