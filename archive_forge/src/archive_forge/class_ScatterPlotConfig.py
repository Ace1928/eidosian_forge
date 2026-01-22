import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class ScatterPlotConfig(ReportAPIBaseModel):
    chart_title: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    z_axis: Optional[str] = None
    x_axis_min: Optional[float] = None
    x_axis_max: Optional[float] = None
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None
    z_axis_min: Optional[float] = None
    z_axis_max: Optional[float] = None
    x_axis_log_scale: Optional[bool] = None
    y_axis_log_scale: Optional[bool] = None
    z_axis_log_scale: Optional[bool] = None
    show_min_y_axis_line: Optional[bool] = None
    show_max_y_axis_line: Optional[bool] = None
    show_avg_y_axis_line: Optional[bool] = None
    legend_template: Optional[str] = None
    custom_gradient: Optional[LList[GradientPoint]] = None
    font_size: Optional[FontSize] = None
    show_linear_regression: Optional[bool] = None