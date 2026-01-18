from test import SHORT_TIMEOUT
from test.with_dummyserver import test_connectionpool
import pytest
import dummyserver.testcase
import urllib3.exceptions
import urllib3.util.retry
import urllib3.util.url
from urllib3.contrib import appengine
urllib3 should retry methods in the default method whitelist