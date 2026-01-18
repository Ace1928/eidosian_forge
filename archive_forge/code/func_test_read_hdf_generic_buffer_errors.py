import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
def test_read_hdf_generic_buffer_errors():
    msg = 'Support for generic buffers has not been implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        read_hdf(BytesIO(b''), 'df')