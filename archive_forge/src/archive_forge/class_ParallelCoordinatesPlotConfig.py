import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class ParallelCoordinatesPlotConfig(ReportAPIBaseModel):
    chart_title: Optional[str] = None
    columns: LList[Column] = Field(default_factory=list)
    custom_gradient: Optional[LList[GradientPoint]] = None
    font_size: Optional[FontSize] = None