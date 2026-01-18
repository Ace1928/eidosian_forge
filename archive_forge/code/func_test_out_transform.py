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
def test_out_transform(self):
    tmpdir = str(self.tmpdir)

    def incr():
        write_text(join(tmpdir, str(uuid4()) + '.txt'), '')
        return len(_LOCAL_FS.glob(join(tmpdir, '*.txt')))

    def t1(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for row in df:
            incr()
            yield row

    def t2(df: pd.DataFrame) -> None:
        incr()

    def t3(df: pd.DataFrame) -> pd.DataFrame:
        incr()
        return df

    @transformer('*', partitionby_has=['b'])
    def t4(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for row in df:
            incr()
            yield row

    @output_transformer(partitionby_has=['b'])
    def t5(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for row in df:
            incr()
            yield row

    class T6(Transformer):

        def get_output_schema(self, df):
            return df.schema

        def transform(self, df):
            incr()
            return df

    class T7(OutputTransformer):

        @property
        def validation_rules(self):
            return {'partitionby_has': 'b'}

        def process(self, df):
            incr()

    def t8(df: pd.DataFrame) -> None:
        incr()
        raise NotImplementedError

    def t9(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
        incr()
        yield df

    def t10(df: pd.DataFrame) -> Iterable[pa.Table]:
        incr()
        yield pa.Table.from_pandas(df)
    with FugueWorkflow() as dag:
        a = dag.df([[1, 2], [3, 4]], 'a:double,b:int')
        a.out_transform(t1)
        a.partition_by('b').out_transform(t2)
        a.partition_by('b').out_transform(t3)
        a.partition_by('b').out_transform(t4)
        a.partition_by('b').out_transform(t5)
        a.out_transform(T6)
        a.partition_by('b').out_transform(T7)
        a.out_transform(t8, ignore_errors=[NotImplementedError])
        a.out_transform(t9)
        a.out_transform(t10)
        raises(FugueWorkflowCompileValidationError, lambda: a.out_transform(t2))
        raises(FugueWorkflowCompileValidationError, lambda: a.out_transform(t3))
        raises(FugueWorkflowCompileValidationError, lambda: a.out_transform(t4))
        raises(FugueWorkflowCompileValidationError, lambda: a.out_transform(t5))
        raises(FugueWorkflowCompileValidationError, lambda: a.out_transform(T7))
    dag.run(self.engine)
    assert 13 <= incr()