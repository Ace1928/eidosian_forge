import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class InlineLink(TextLikeMixin, InlineModel):
    type: Literal['link'] = 'link'
    url: str = 'https://'
    children: LList[Text] = Field(default_factory=lambda: [Text()])