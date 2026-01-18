import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
@dec.skip_without('matplotlib')
def test_set_matplotlib_formats():
    from matplotlib.figure import Figure
    formatters = get_ipython().display_formatter.formatters
    for formats in [('png',), ('pdf', 'svg'), ('jpeg', 'retina', 'png'), ()]:
        active_mimes = {_fmt_mime_map[fmt] for fmt in formats}
        with pytest.deprecated_call():
            display.set_matplotlib_formats(*formats)
        for mime, f in formatters.items():
            if mime in active_mimes:
                assert Figure in f
            else:
                assert Figure not in f