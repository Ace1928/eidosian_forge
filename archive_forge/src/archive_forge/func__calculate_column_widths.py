from dataclasses import dataclass, field, replace
from typing import (
from . import box, errors
from ._loop import loop_first_last, loop_last
from ._pick import pick_bool
from ._ratio import ratio_distribute, ratio_reduce
from .align import VerticalAlignMethod
from .jupyter import JupyterMixin
from .measure import Measurement
from .padding import Padding, PaddingDimensions
from .protocol import is_renderable
from .segment import Segment
from .style import Style, StyleType
from .text import Text, TextType
def _calculate_column_widths(self, console: 'Console', options: 'ConsoleOptions') -> List[int]:
    """Calculate the widths of each column, including padding, not including borders."""
    max_width = options.max_width
    columns = self.columns
    width_ranges = [self._measure_column(console, options, column) for column in columns]
    widths = [_range.maximum or 1 for _range in width_ranges]
    get_padding_width = self._get_padding_width
    extra_width = self._extra_width
    if self.expand:
        ratios = [col.ratio or 0 for col in columns if col.flexible]
        if any(ratios):
            fixed_widths = [0 if column.flexible else _range.maximum for _range, column in zip(width_ranges, columns)]
            flex_minimum = [(column.width or 1) + get_padding_width(column._index) for column in columns if column.flexible]
            flexible_width = max_width - sum(fixed_widths)
            flex_widths = ratio_distribute(flexible_width, ratios, flex_minimum)
            iter_flex_widths = iter(flex_widths)
            for index, column in enumerate(columns):
                if column.flexible:
                    widths[index] = fixed_widths[index] + next(iter_flex_widths)
    table_width = sum(widths)
    if table_width > max_width:
        widths = self._collapse_widths(widths, [column.width is None and (not column.no_wrap) for column in columns], max_width)
        table_width = sum(widths)
        if table_width > max_width:
            excess_width = table_width - max_width
            widths = ratio_reduce(excess_width, [1] * len(widths), widths, widths)
            table_width = sum(widths)
        width_ranges = [self._measure_column(console, options.update_width(width), column) for width, column in zip(widths, columns)]
        widths = [_range.maximum or 0 for _range in width_ranges]
    if table_width < max_width and self.expand or (self.min_width is not None and table_width < self.min_width - extra_width):
        _max_width = max_width if self.min_width is None else min(self.min_width - extra_width, max_width)
        pad_widths = ratio_distribute(_max_width - table_width, widths)
        widths = [_width + pad for _width, pad in zip(widths, pad_widths)]
    return widths