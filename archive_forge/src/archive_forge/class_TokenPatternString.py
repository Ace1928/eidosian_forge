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
class TokenPatternString(BaseModel):
    REGEX: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='regex')
    IN: Optional[List[StrictStr]] = Field(None, alias='in')
    NOT_IN: Optional[List[StrictStr]] = Field(None, alias='not_in')
    IS_SUBSET: Optional[List[StrictStr]] = Field(None, alias='is_subset')
    IS_SUPERSET: Optional[List[StrictStr]] = Field(None, alias='is_superset')
    INTERSECTS: Optional[List[StrictStr]] = Field(None, alias='intersects')
    FUZZY: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy')
    FUZZY1: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy1')
    FUZZY2: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy2')
    FUZZY3: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy3')
    FUZZY4: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy4')
    FUZZY5: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy5')
    FUZZY6: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy6')
    FUZZY7: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy7')
    FUZZY8: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy8')
    FUZZY9: Optional[Union[StrictStr, 'TokenPatternString']] = Field(None, alias='fuzzy9')

    class Config:
        extra = 'forbid'
        allow_population_by_field_name = True

    @validator('*', pre=True, each_item=True, allow_reuse=True)
    def raise_for_none(cls, v):
        if v is None:
            raise ValueError('None / null is not allowed')
        return v