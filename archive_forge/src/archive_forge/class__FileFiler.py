import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import pyarrow as pa
import ray.data as rd
from pyarrow import csv as pacsv
from pyarrow import json as pajson
from ray.data.datasource import FileExtensionFilter
from triad.collections import Schema
from triad.collections.dict import ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.io import exists, makedirs, rm
from fugue import ExecutionEngine
from fugue._utils.io import FileParser, save_df
from fugue.collections.partition import PartitionSpec
from fugue.dataframe import DataFrame
from fugue_ray.dataframe import RayDataFrame
class _FileFiler(FileExtensionFilter):

    def __init__(self, file_extensions: Union[str, List[str]], exclude: Iterable[str]):
        super().__init__(file_extensions, allow_if_no_extension=True)
        self._exclude = set(exclude)

    def _is_valid(self, path: str) -> bool:
        return pathlib.Path(path).name not in self._exclude and self._file_has_extension(path)

    def __call__(self, paths: List[str]) -> List[str]:
        return [path for path in paths if self._is_valid(path)]