from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_parse_join_types(self):
    assert 'cross' == self.op.pl_utils.parse_join_type('CROss')
    assert 'inner' == self.op.pl_utils.parse_join_type('join')
    assert 'inner' == self.op.pl_utils.parse_join_type('Inner')
    assert 'left_outer' == self.op.pl_utils.parse_join_type('left')
    assert 'left_outer' == self.op.pl_utils.parse_join_type('left  outer')
    assert 'right_outer' == self.op.pl_utils.parse_join_type('right')
    assert 'right_outer' == self.op.pl_utils.parse_join_type('right_ outer')
    assert 'full_outer' == self.op.pl_utils.parse_join_type('full')
    assert 'full_outer' == self.op.pl_utils.parse_join_type(' outer ')
    assert 'full_outer' == self.op.pl_utils.parse_join_type('full_outer')
    assert 'left_anti' == self.op.pl_utils.parse_join_type('anti')
    assert 'left_anti' == self.op.pl_utils.parse_join_type('left anti')
    assert 'left_semi' == self.op.pl_utils.parse_join_type('semi')
    assert 'left_semi' == self.op.pl_utils.parse_join_type('left semi')
    raises(NotImplementedError, lambda: self.op.pl_utils.parse_join_type('right semi'))