from __future__ import division, print_function, absolute_import
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xls import fromxls, toxls
from petl.test.helpers import ieq
def test_passing_kwargs_to_xlutils_view():
    filename = _get_test_xls()
    if filename is None:
        return
    from petl.io.xlutils_view import View
    org_init = View.__init__

    def wrapper(self, *args, **kwargs):
        assert 'ignore_workbook_corruption' in kwargs
        return org_init(self, *args, **kwargs)
    with patch('petl.io.xlutils_view.View.__init__', wrapper):
        tbl = fromxls(filename, 'Sheet1', use_view=True, ignore_workbook_corruption=True)
        expect = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 2), (u'é', datetime(2012, 1, 1)))
        ieq(expect, tbl)
        ieq(expect, tbl)