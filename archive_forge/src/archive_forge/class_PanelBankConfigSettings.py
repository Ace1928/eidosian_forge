import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class PanelBankConfigSettings(ReportAPIBaseModel):
    auto_organize_prefix: int = 2
    show_empty_sections: bool = False
    sort_alphabetically: bool = False