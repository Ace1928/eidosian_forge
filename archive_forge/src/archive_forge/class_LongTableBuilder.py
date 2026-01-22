from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class LongTableBuilder(GenericTableBuilder):
    """Concrete table builder for longtable.

    >>> from pandas.io.formats import format as fmt
    >>> df = pd.DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = LongTableBuilder(formatter, caption='a long table',
    ...                            label='tab:long', column_format='lrl')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{longtable}{lrl}
    \\caption{a long table}
    \\label{tab:long}\\\\
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    \\endfirsthead
    \\caption[]{a long table} \\\\
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    \\endhead
    \\midrule
    \\multicolumn{3}{r}{{Continued on next page}} \\\\
    \\midrule
    \\endfoot
    <BLANKLINE>
    \\bottomrule
    \\endlastfoot
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\end{longtable}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str:
        first_row = f'\\begin{{longtable}}{self._position_macro}{{{self.column_format}}}'
        elements = [first_row, f'{self._caption_and_label()}']
        return '\n'.join([item for item in elements if item])

    def _caption_and_label(self) -> str:
        if self.caption or self.label:
            double_backslash = '\\\\'
            elements = [f'{self._caption_macro}', f'{self._label_macro}']
            caption_and_label = '\n'.join([item for item in elements if item])
            caption_and_label += double_backslash
            return caption_and_label
        else:
            return ''

    @property
    def middle_separator(self) -> str:
        iterator = self._create_row_iterator(over='header')
        elements = ['\\midrule', '\\endfirsthead', f'\\caption[]{{{self.caption}}} \\\\' if self.caption else '', self.top_separator, self.header, '\\midrule', '\\endhead', '\\midrule', f'\\multicolumn{{{len(iterator.strcols)}}}{{r}}{{{{Continued on next page}}}} \\\\', '\\midrule', '\\endfoot\n', '\\bottomrule', '\\endlastfoot']
        if self._is_separator_required():
            return '\n'.join(elements)
        return ''

    @property
    def bottom_separator(self) -> str:
        return ''

    @property
    def env_end(self) -> str:
        return '\\end{longtable}'