import sys
from pathlib import Path
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
class FileValidationError(ValueError):
    """Error for paths outside the root directory."""