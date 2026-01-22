from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
def _create_idx_iter(self, n_cols, n_rows):
    col_indices = range(n_cols) if self._fills_from_left_to_right else range(n_cols - 1, -1, -1)
    row_indices = range(n_rows) if self._fills_from_top_to_bottom else range(n_rows - 1, -1, -1)
    if self._fills_row_first:
        return ((col_index, row_index) for row_index, col_index in product(row_indices, col_indices))
    else:
        return product(col_indices, row_indices)