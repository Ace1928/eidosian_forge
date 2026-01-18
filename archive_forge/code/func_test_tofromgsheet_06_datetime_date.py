from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
@found_gcp_credentials
@pytest.mark.xfail(raises=TypeError, reason='When this stop failing, uncomment datetime.date in TEST1 and TEST2')
def test_tofromgsheet_06_datetime_date():
    test_table_dt = [[x[0], datetime.date(2012, 5, 6)] if i in [5] else x for i, x in enumerate(TEST_TABLE[:])]
    _test_to_fromg_sheet(test_table_dt[:], None, 'B1:B4', test_table_dt[:])