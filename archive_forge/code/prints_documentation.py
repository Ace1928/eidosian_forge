import os
import warnings
from functools import partial, wraps
from typing import Any, Callable
from torchmetrics import _logger as log
Warn user that he is importing function from location it has been deprecated.