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
def test_tofromgsheet_03_empty_table():
    _test_to_fromg_sheet((), None, None, ())