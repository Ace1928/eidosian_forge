import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class CalloutLine(ReportAPIBaseModel):
    type: Literal['callout-line'] = 'callout-line'
    children: LList[TextLikeInternal] = Field(default_factory=lambda: [Text()])