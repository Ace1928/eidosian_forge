import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pyarrow as pa
@st.composite
def _pylist(draw, value_type, size, nullable=True):
    arr = draw(arrays(value_type, size=size, nullable=False))
    return arr.to_pylist()