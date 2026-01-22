import inspect
import re
from collections import defaultdict
from enum import Enum
from typing import (
from thinc.api import ConfigValidationError, Model, Optimizer
from thinc.config import Promise
from .attrs import NAMES
from .compat import Literal
from .lookups import Lookups
from .util import is_cython_func
class DocJSONSchema(BaseModel):
    """
    JSON/dict format for JSON representation of Doc objects.
    """
    cats: Optional[Dict[StrictStr, StrictFloat]] = Field(None, title='Categories with corresponding probabilities')
    ents: Optional[List[Dict[StrictStr, Union[StrictInt, StrictStr]]]] = Field(None, title='Information on entities')
    sents: Optional[List[Dict[StrictStr, StrictInt]]] = Field(None, title="Indices of sentences' start and end indices")
    text: StrictStr = Field(..., title='Document text')
    spans: Optional[Dict[StrictStr, List[Dict[StrictStr, Union[StrictStr, StrictInt]]]]] = Field(None, title='Span information - end/start indices, label, KB ID')
    tokens: List[Dict[StrictStr, Union[StrictStr, StrictInt]]] = Field(..., title='Token information - ID, start, annotations')
    underscore_doc: Optional[Dict[StrictStr, Any]] = Field(None, title="Any custom data stored in the document's _ attribute", alias='_')
    underscore_token: Optional[Dict[StrictStr, List[Dict[StrictStr, Any]]]] = Field(None, title="Any custom data stored in the token's _ attribute")
    underscore_span: Optional[Dict[StrictStr, List[Dict[StrictStr, Any]]]] = Field(None, title="Any custom data stored in the span's _ attribute")