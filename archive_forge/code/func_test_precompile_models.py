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
def test_precompile_models(eight_schools_params, draws, chains):
    """Precompile model files."""
    load_cached_models(eight_schools_params, draws, chains)