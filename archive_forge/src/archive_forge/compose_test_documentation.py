import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
Tests automatic removal of initializers when merging graphs