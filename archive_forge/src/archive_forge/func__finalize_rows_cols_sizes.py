from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
def _finalize_rows_cols_sizes(self):
    selfw = self.width
    selfh = self.height
    if self.col_force_default:
        cols = [self.col_default_width] * len(self._cols)
        for index, value in self.cols_minimum.items():
            cols[index] = value
        self._cols = cols
    else:
        cols = self._cols
        cols_sh = self._cols_sh
        cols_sh_min = self._cols_sh_min
        cols_weight = float(sum((x for x in cols_sh if x is not None)))
        stretch_w = max(0.0, selfw - self._cols_min_size_none)
        if stretch_w > 1e-09:
            if self._has_hint_bound_x:
                self.layout_hint_with_bounds(cols_weight, stretch_w, sum((c for c in cols_sh_min if c is not None)), cols_sh_min, self._cols_sh_max, cols_sh)
            for index, col_stretch in enumerate(cols_sh):
                if not col_stretch:
                    continue
                cols[index] += stretch_w * col_stretch / cols_weight
    if self.row_force_default:
        rows = [self.row_default_height] * len(self._rows)
        for index, value in self.rows_minimum.items():
            rows[index] = value
        self._rows = rows
    else:
        rows = self._rows
        rows_sh = self._rows_sh
        rows_sh_min = self._rows_sh_min
        rows_weight = float(sum((x for x in rows_sh if x is not None)))
        stretch_h = max(0.0, selfh - self._rows_min_size_none)
        if stretch_h > 1e-09:
            if self._has_hint_bound_y:
                self.layout_hint_with_bounds(rows_weight, stretch_h, sum((r for r in rows_sh_min if r is not None)), rows_sh_min, self._rows_sh_max, rows_sh)
            for index, row_stretch in enumerate(rows_sh):
                if not row_stretch:
                    continue
                rows[index] += stretch_h * row_stretch / rows_weight