import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class Column(ReportAPIBaseModel):
    accessor: str
    display_name: Optional[str] = None
    inverted: Optional[bool] = None
    log: Optional[bool] = None
    ref: Optional[Ref] = None