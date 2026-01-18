import time
import logging
import typing as tp
from IPython.display import display
from copy import deepcopy
from typing import List, Optional, Any, Union
from .ipythonwidget import MetricWidget
def update_data(self, data: tp.Dict) -> None:
    self.data = deepcopy(data)