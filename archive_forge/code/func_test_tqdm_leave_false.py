import time
import numpy as np
import pandas as pd
import pytest
from tqdm.contrib.concurrent import process_map
import panel as pn
from panel.widgets import Tqdm
def test_tqdm_leave_false():
    tqdm = Tqdm(layout='row', sizing_mode='stretch_width')
    for _ in tqdm(range(3), leave=False):
        pass
    assert tqdm.value == 0
    assert tqdm.max == 3
    assert tqdm.text == ''