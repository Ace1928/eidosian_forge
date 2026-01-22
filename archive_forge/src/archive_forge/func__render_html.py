from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def _render_html(self, sparse_index: bool, sparse_columns: bool, max_rows: int | None=None, max_cols: int | None=None, **kwargs) -> str:
    """
        Renders the ``Styler`` including all applied styles to HTML.
        Generates a dict with necessary kwargs passed to jinja2 template.
        """
    d = self._render(sparse_index, sparse_columns, max_rows, max_cols, '&nbsp;')
    d.update(kwargs)
    return self.template_html.render(**d, html_table_tpl=self.template_html_table, html_style_tpl=self.template_html_style)