import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
class RandomVariableTestClass:
    """Example class for random variables."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        """Return argument to constructor as string representation."""
        return self.name