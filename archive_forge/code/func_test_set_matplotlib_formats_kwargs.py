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
def test_set_matplotlib_formats_kwargs():
    from matplotlib.figure import Figure
    ip = get_ipython()
    cfg = _get_inline_config()
    cfg.print_figure_kwargs.update(dict(foo='bar'))
    kwargs = dict(dpi=150)
    with pytest.deprecated_call():
        display.set_matplotlib_formats('png', **kwargs)
    formatter = ip.display_formatter.formatters['image/png']
    f = formatter.lookup_by_type(Figure)
    formatter_kwargs = f.keywords
    expected = kwargs
    expected['base64'] = True
    expected['fmt'] = 'png'
    expected.update(cfg.print_figure_kwargs)
    assert formatter_kwargs == expected