import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class Filters(ReportAPIBaseModel):
    op: Ops = 'OR'
    key: Optional[Key] = None
    filters: Optional[LList['Filters']] = None
    value: Optional[Any] = None
    disabled: Optional[bool] = None