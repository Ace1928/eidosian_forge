from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
def to_request_kwargs(self, exclude_none: bool=True, exclude_unset: bool=True, **kwargs) -> Dict[str, Any]:
    """
        Returns the request kwargs
        """
    return self.model_dump(include=set(self.callback_param_fields), exclude_none=exclude_none, exclude_unset=exclude_unset, **kwargs)