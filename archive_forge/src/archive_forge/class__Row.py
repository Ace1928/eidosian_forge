import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
class _Row:

    def __init__(self, results, row_group, render_env, env_str_len, row_name_str_len, time_scale, colorize, num_threads=None):
        super().__init__()
        self._results = results
        self._row_group = row_group
        self._render_env = render_env
        self._env_str_len = env_str_len
        self._row_name_str_len = row_name_str_len
        self._time_scale = time_scale
        self._colorize = colorize
        self._columns: Tuple[_Column, ...] = ()
        self._num_threads = num_threads

    def register_columns(self, columns: Tuple[_Column, ...]):
        self._columns = columns

    def as_column_strings(self):
        concrete_results = [r for r in self._results if r is not None]
        env = f'({concrete_results[0].env})' if self._render_env else ''
        env = env.ljust(self._env_str_len + 4)
        output = ['  ' + env + concrete_results[0].as_row_name]
        for m, col in zip(self._results, self._columns or ()):
            if m is None:
                output.append(col.num_to_str(None, 1, None))
            else:
                output.append(col.num_to_str(m.median / self._time_scale, m.significant_figures, m.iqr / m.median if m.has_warnings else None))
        return output

    @staticmethod
    def color_segment(segment, value, best_value):
        if value <= best_value * 1.01 or value <= best_value + 1e-07:
            return BEST + BOLD + segment + TERMINATE * 2
        if value <= best_value * 1.1:
            return GOOD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 5:
            return VERY_BAD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 2:
            return BAD + segment + TERMINATE * 2
        return segment

    def row_separator(self, overall_width):
        return [f'{self._num_threads} threads: '.ljust(overall_width, '-')] if self._num_threads is not None else []

    def finalize_column_strings(self, column_strings, col_widths):
        best_values = [-1 for _ in column_strings]
        if self._colorize == Colorize.ROWWISE:
            row_min = min((r.median for r in self._results if r is not None))
            best_values = [row_min for _ in column_strings]
        elif self._colorize == Colorize.COLUMNWISE:
            best_values = [optional_min((r.median for r in column.get_results_for(self._row_group) if r is not None)) for column in self._columns or ()]
        row_contents = [column_strings[0].ljust(col_widths[0])]
        for col_str, width, result, best_value in zip(column_strings[1:], col_widths[1:], self._results, best_values):
            col_str = col_str.center(width)
            if self._colorize != Colorize.NONE and result is not None and (best_value is not None):
                col_str = self.color_segment(col_str, result.median, best_value)
            row_contents.append(col_str)
        return row_contents