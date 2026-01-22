import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class LinePlotConfig(ReportAPIBaseModel):
    chart_title: Optional[str] = None
    x_axis: Optional[str] = None
    metrics: LList[str] = Field(default_factory=list)
    x_axis_min: Optional[float] = None
    x_axis_max: Optional[float] = None
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None
    x_log_scale: Optional[bool] = None
    y_log_scale: Optional[bool] = None
    x_axis_title: Optional[str] = None
    y_axis_title: Optional[str] = None
    ignore_outliers: Optional[bool] = None
    group_by: Optional[str] = None
    group_agg: Optional[GroupAgg] = None
    group_area: Optional[GroupArea] = None
    smoothing_weight: Optional[float] = None
    smoothing_type: Optional[SmoothingType] = None
    show_original_after_smoothing: Optional[bool] = None
    limit: Optional[int] = None
    expressions: Optional[LList[str]] = None
    plot_type: Optional[LinePlotStyle] = None
    font_size: Optional[FontSize] = None
    legend_position: Optional[LegendPosition] = None
    legend_template: Optional[str] = None
    aggregate: Optional[bool] = None
    x_expression: Optional[str] = None
    override_line_widths: Optional[dict] = None
    override_colors: Optional[dict] = None
    override_series_titles: Optional[dict] = None
    legend_fields: Optional[list] = None