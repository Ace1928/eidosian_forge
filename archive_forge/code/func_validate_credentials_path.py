import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
@validator('credentials_path')
def validate_credentials_path(cls, v: Any, **kwargs: Any) -> Any:
    """Validate that credentials_path exists."""
    if not v.exists():
        raise ValueError(f'credentials_path {v} does not exist')
    return v