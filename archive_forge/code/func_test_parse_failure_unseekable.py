from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def test_parse_failure_unseekable(self, flavor_read_html):
    if flavor_read_html.keywords.get('flavor') == 'lxml':
        pytest.skip('Not applicable for lxml')

    class UnseekableStringIO(StringIO):

        def seekable(self):
            return False
    bad = UnseekableStringIO('\n            <table><tr><td>spam<foobr />eggs</td></tr></table>')
    assert flavor_read_html(bad)
    with pytest.raises(ValueError, match='passed a non-rewindable file object'):
        flavor_read_html(bad)