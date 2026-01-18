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
def test_tofromgsheet_07_open_by_key():
    filename, gspread_client, emails = _get_gspread_test_params()
    table = TEST_TABLE[:]
    spread_id = togsheet(table, gspread_client, filename, share_emails=emails)
    try:
        result = fromgsheet(gspread_client, spread_id, open_by_key=True)
        ieq(table, result)
    finally:
        gspread_client.del_spreadsheet(spread_id)