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
def _render_latex(self, sparse_index: bool, sparse_columns: bool, clines: str | None, **kwargs) -> str:
    """
        Render a Styler in latex format
        """
    d = self._render(sparse_index, sparse_columns, None, None)
    self._translate_latex(d, clines=clines)
    self.template_latex.globals['parse_wrap'] = _parse_latex_table_wrapping
    self.template_latex.globals['parse_table'] = _parse_latex_table_styles
    self.template_latex.globals['parse_cell'] = _parse_latex_cell_styles
    self.template_latex.globals['parse_header'] = _parse_latex_header_span
    d.update(kwargs)
    return self.template_latex.render(**d)