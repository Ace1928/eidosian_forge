import time
import logging
import typing as tp
from IPython.display import display
from copy import deepcopy
from typing import List, Optional, Any, Union
from .ipythonwidget import MetricWidget

        Save metrics at specific training epoch.

        Parameters
        ----------
        epoch : int
            Current epoch

        train : bool
            Flag that indicates whether metrics are calculated on train or test data

        metrics: dict
            Values for each of metrics defined in `__init__` method of this class
        