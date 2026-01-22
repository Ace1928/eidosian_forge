from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def _write_table(self, indent: int=0) -> None:
    _classes = ['dataframe']
    use_mathjax = get_option('display.html.use_mathjax')
    if not use_mathjax:
        _classes.append('tex2jax_ignore')
    if self.classes is not None:
        if isinstance(self.classes, str):
            self.classes = self.classes.split()
        if not isinstance(self.classes, (list, tuple)):
            raise TypeError(f'classes must be a string, list, or tuple, not {type(self.classes)}')
        _classes.extend(self.classes)
    if self.table_id is None:
        id_section = ''
    else:
        id_section = f' id="{self.table_id}"'
    if self.border is None:
        border_attr = ''
    else:
        border_attr = f' border="{self.border}"'
    self.write(f'<table{border_attr} class="{' '.join(_classes)}"{id_section}>', indent)
    if self.fmt.header or self.show_row_idx_names:
        self._write_header(indent + self.indent_delta)
    self._write_body(indent + self.indent_delta)
    self.write('</table>', indent)