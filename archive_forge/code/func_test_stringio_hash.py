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
def test_stringio_hash():
    sio1, sio2 = (io.StringIO(), io.StringIO())
    sio1.write('foo')
    sio2.write('foo')
    sio1.seek(0)
    sio2.seek(0)
    assert hashes_equal(sio1, sio2)
    sio3 = io.StringIO()
    sio3.write('bar')
    sio3.seek(0)
    assert not hashes_equal(sio1, sio3)