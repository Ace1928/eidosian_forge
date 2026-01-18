import gzip
import http.server
from io import BytesIO
import multiprocessing
import socket
import time
import urllib.error
import pytest
from pandas.compat import is_ci_environment
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def write_back_bytes(self, response_bytes):
    """
        shared logic at the end of a GET request
        """
    self.wfile.write(response_bytes)