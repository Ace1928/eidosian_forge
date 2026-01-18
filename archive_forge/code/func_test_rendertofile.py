import pytest
import contextlib
import os
import tempfile
from rpy2.robjects.packages import importr, data
from rpy2.robjects import r
from rpy2.robjects.lib import grdevices
def test_rendertofile():
    fn = tempfile.mktemp(suffix='.png')
    with set_filenames_to_delete() as todelete:
        todelete.add(fn)
        with grdevices.render_to_file(grdevices.png, filename=fn) as d:
            r(' plot(0) ')
        assert os.path.exists(fn)