from __future__ import annotations
from dataclasses import dataclass, fields, field
import textwrap
from typing import Any, Callable, Union
from collections.abc import Generator
import numpy as np
import pandas as pd
import matplotlib as mpl
from numpy import ndarray
from pandas import DataFrame
from matplotlib.artist import Artist
from seaborn._core.scales import Scale
from seaborn._core.properties import (
from seaborn._core.exceptions import PlotSpecError
@property
def _grouping_props(self):
    return [f.name for f in fields(self) if isinstance(f.default, Mappable) and f.default.grouping]