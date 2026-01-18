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
def test_outputter_validation(self):

    def o1(dfs: DataFrames) -> None:
        pass

    @outputter(input_has=['a'])
    def o2(dfs: DataFrames) -> None:
        pass

    class O3(Outputter):

        @property
        def validation_rules(self):
            return dict(input_has=['a'])

        def process(self, dfs: DataFrames) -> None:
            pass
    for o in [o1, o2, O3]:
        with raises(FugueWorkflowRuntimeValidationError):
            with FugueWorkflow() as dag:
                df1 = dag.df([[0, 1]], 'a:int,b:int')
                df2 = dag.df([[0, 1]], 'c:int,d:int')
                dag.output([df1, df2], using=o)
            dag.run(self.engine)
        with FugueWorkflow() as dag:
            df1 = dag.df([[0, 1]], 'a:int,b:int')
            df2 = dag.df([[0, 1]], 'a:int,b:int')
            dag.output([df1, df2], using=o)
        dag.run(self.engine)

    def o4(dfs: DataFrames) -> None:
        pass
    with raises(FugueWorkflowCompileValidationError):
        dag = FugueWorkflow()
        df = dag.df([[0, 1]], 'a:int,b:int')
        df.output(o4)
    with FugueWorkflow() as dag:
        dag.df([[0, 1]], 'a:int,b:int').partition(by=['b']).output(o4)
    dag.run(self.engine)