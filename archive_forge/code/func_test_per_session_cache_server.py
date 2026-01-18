import datetime as dt
import io
import pathlib
import time
from collections import Counter
import numpy as np
import pandas as pd
import param
import pytest
import requests
from panel.io.cache import _find_hash_func, cache
from panel.io.state import set_curdoc, state
from panel.tests.util import serve_and_wait
def test_per_session_cache_server(port):
    counts = Counter()

    @cache(per_session=True)
    def get_data():
        counts[state.curdoc] += 1
        return 'Some data'

    def app():
        get_data()
        get_data()
        return
    serve_and_wait(app, port=port)
    requests.get(f'http://localhost:{port}/')
    requests.get(f'http://localhost:{port}/')
    assert list(counts.values()) == [1, 1]