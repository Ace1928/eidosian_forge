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
def test_yield_table(self):
    with raises(FugueWorkflowCompileError):
        FugueWorkflow().df([[0]], 'a:int').checkpoint().yield_table_as('x')
    with raises(ValueError):
        FugueWorkflow().df([[0]], 'a:int').persist().yield_table_as('x')

    def run_test(deterministic):
        dag1 = FugueWorkflow()
        df = dag1.df([[0]], 'a:int')
        if deterministic:
            df = df.deterministic_checkpoint(storage_type='table')
        df.yield_table_as('x')
        id1 = dag1.spec_uuid()
        dag2 = FugueWorkflow()
        dag2.df([[0]], 'a:int').assert_eq(dag2.df(dag1.yields['x']))
        id2 = dag2.spec_uuid()
        dag1.run(self.engine)
        dag2.run(self.engine)
        return (id1, id2)
    id1, id2 = run_test(False)
    id3, id4 = run_test(False)
    assert id1 == id3
    assert id2 != id4
    id1, id2 = run_test(True)
    id3, id4 = run_test(True)
    assert id1 == id3
    assert id2 == id4