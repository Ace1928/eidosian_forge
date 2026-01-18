import time
import numpy as np
import pandas as pd
import pytest
from tqdm.contrib.concurrent import process_map
import panel as pn
from panel.widgets import Tqdm
def test_process_map():
    pytest.skip('Skip due to issues pickling callers on Parameterized objects.')
    tqdm_obj = Tqdm()
    assert tqdm_obj.value == 0
    NUM_ITEMS = 10
    _ = process_map(time.sleep, [0.3] * NUM_ITEMS, max_workers=2, tqdm_class=tqdm_obj)
    assert tqdm_obj.value == NUM_ITEMS