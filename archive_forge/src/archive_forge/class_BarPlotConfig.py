import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class BarPlotConfig(ReportAPIBaseModel):
    chart_title: Optional[str] = None
    metrics: LList[str] = Field(default_factory=list)
    vertical: bool = False
    x_axis_min: Optional[float] = None
    x_axis_max: Optional[float] = None
    x_axis_title: Optional[str] = None
    y_axis_title: Optional[str] = None
    group_by: Optional[str] = None
    group_agg: Optional[GroupAgg] = None
    group_area: Optional[GroupArea] = None
    limit: Optional[int] = None
    bar_limit: Optional[int] = None
    expressions: Optional[LList[str]] = None
    legend_template: Optional[str] = None
    font_size: Optional[FontSize] = None
    override_series_titles: Optional[dict] = None
    override_colors: Optional[dict] = None