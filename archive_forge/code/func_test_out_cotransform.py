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
def test_out_cotransform(self):
    tmpdir = str(self.tmpdir)

    def incr():
        write_text(join(tmpdir, str(uuid4()) + '.txt'), '')
        return len(_LOCAL_FS.glob(join(tmpdir, '*.txt')))

    def t1(df: Iterable[Dict[str, Any]], df2: pd.DataFrame) -> Iterable[Dict[str, Any]]:
        for row in df:
            incr()
            yield row

    def t2(dfs: DataFrames) -> None:
        incr()

    def t3(df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        incr()
        return df

    @cotransformer('a:double,b:int')
    def t4(df: Iterable[Dict[str, Any]], df2: pd.DataFrame) -> Iterable[Dict[str, Any]]:
        for row in df:
            incr()
            yield row

    @output_cotransformer()
    def t5(df: Iterable[Dict[str, Any]], df2: pd.DataFrame) -> Iterable[Dict[str, Any]]:
        for row in df:
            incr()
            yield row

    class T6(CoTransformer):

        def get_output_schema(self, dfs):
            return dfs[0].schema

        def transform(self, dfs):
            incr()
            return dfs[0]

    class T7(OutputCoTransformer):

        def process(self, dfs):
            incr()

    def t8(df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        incr()
        raise NotImplementedError

    def t9(df1: pd.DataFrame, df2: pd.DataFrame) -> Iterable[pd.DataFrame]:
        incr()
        for df in [df1, df2]:
            yield df
    with FugueWorkflow() as dag:
        a0 = dag.df([[1, 2], [3, 4]], 'a:double,b:int')
        a1 = dag.df([[1, 2], [3, 4]], 'aa:double,b:int')
        a = dag.zip(a0, a1)
        a.out_transform(t1)
        a.out_transform(t2)
        a.out_transform(t3)
        a.out_transform(t4)
        b = dag.zip(dict(df=a0, df2=a1))
        b.out_transform(t5)
        a.out_transform(T6)
        a.out_transform(T7)
        a.out_transform(t8, ignore_errors=[NotImplementedError])
        a.out_transform(t9)
    dag.run(self.engine)
    assert 12 <= incr()