import io
import pickle
from typing import Union, Any
from pydantic import BaseModel
from pydantic.types import ByteSize
class SizeModel(BaseModel):
    size: ByteSize