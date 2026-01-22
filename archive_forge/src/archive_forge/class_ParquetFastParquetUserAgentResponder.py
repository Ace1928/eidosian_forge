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
class ParquetFastParquetUserAgentResponder(BaseUserAgentResponder):

    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        import fsspec
        response_df.to_parquet('memory://fastparquet_user_agent.parquet', index=False, engine='fastparquet', compression=None)
        with fsspec.open('memory://fastparquet_user_agent.parquet', 'rb') as f:
            response_bytes = f.read()
        self.write_back_bytes(response_bytes)