from __future__ import annotations
import os
from typing import (
import param
from param.parameterized import register_reference_transform
from pyviz_comms import JupyterComm
from ..config import config
from ..models import IPyWidget as _BkIPyWidget
from .base import PaneBase

    Transforms an ipywidget into a Parameter that listens updates
    when the ipywidget updates.
    