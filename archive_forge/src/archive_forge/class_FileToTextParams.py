import os
from fastapi import (
from pydantic import BaseModel
from typing import Iterator
class FileToTextParams(BaseModel):
    file_name: str
    file_encoding: str = 'utf-8'