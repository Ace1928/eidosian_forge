from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
def _get_gspread_test_params():
    filename = 'test-{}'.format(str(uuid.uuid4()))
    gspread_client = _get_gspread_client()
    emails = _get_env_sharing_emails()
    return (filename, gspread_client, emails)