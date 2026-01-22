import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class GalleryLinkURL(ReportAPIBaseModel):
    type: Literal['url'] = 'url'
    url: str = ''
    title: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = Field(..., alias='imageURL')