import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
class MockCoTransform1(CoTransformer):

    def get_output_schema(self, dfs: DataFrames) -> Any:
        assert 'test' in self.workflow_conf
        assert 2 == len(dfs)
        if self.params.get('named', False):
            assert dfs.has_key
        else:
            assert not dfs.has_key
        return [self.key_schema, 'ct1:int,ct2:int,p:int']

    def on_init(self, dfs: DataFrames) -> None:
        assert 'test' in self.workflow_conf
        assert 2 == len(dfs)
        if self.params.get('named', False):
            assert dfs.has_key
        else:
            assert not dfs.has_key
        self.pn = self.cursor.physical_partition_no
        self.ks = self.key_schema
        if 'on_init_called' not in self.__dict__:
            self.on_init_called = 1
        else:
            self.on_init_called += 1

    def transform(self, dfs: DataFrames) -> LocalDataFrame:
        assert 1 == self.on_init_called
        assert 'test' in self.workflow_conf
        assert 2 == len(dfs)
        if self.params.get('named', False):
            assert dfs.has_key
        else:
            assert not dfs.has_key
        row = self.cursor.key_value_array + [dfs[0].count(), dfs[1].count(), self.params.get('p', 1)]
        return ArrayDataFrame([row], self.output_schema)