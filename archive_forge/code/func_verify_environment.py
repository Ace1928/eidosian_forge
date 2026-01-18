import json
import os
from google.auth import _helpers
import google.auth.transport.requests
import google.auth.transport.urllib3
import pytest
import pytest_asyncio
import requests
import urllib3
import aiohttp
from google.auth.transport import _aiohttp_requests as aiohttp_requests
from system_tests.system_tests_sync import conftest as sync_conftest
def verify_environment():
    """Checks to make sure that requisite data files are available."""
    if not os.path.isdir(sync_conftest.DATA_DIR):
        raise EnvironmentError('In order to run system tests, test data must exist in system_tests/data. See CONTRIBUTING.rst for details.')