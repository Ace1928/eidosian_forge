from __future__ import annotations
import pathlib
import secrets
import shutil
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from fastapi import Request
from gradio_client.utils import traverse
from . import wasm_utils
class GradioBaseModel(ABC):

    def copy_to_dir(self, dir: str | pathlib.Path) -> GradioDataModel:
        if not isinstance(self, (BaseModel, RootModel)):
            raise TypeError('must be used in a Pydantic model')
        dir = pathlib.Path(dir)

        def unique_copy(obj: dict):
            data = FileData(**obj)
            return data._copy_to_dir(str(pathlib.Path(dir / secrets.token_hex(10)))).model_dump()
        return self.__class__.from_json(x=traverse(self.model_dump(), unique_copy, FileData.is_file_data))

    @classmethod
    @abstractmethod
    def from_json(cls, x) -> GradioDataModel:
        pass