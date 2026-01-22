import itertools
from kivy.uix.recyclelayout import RecycleLayout
from kivy.uix.gridlayout import GridLayout, GridLayoutException, nmax, nmin
from collections import defaultdict
def _fill_rows_cols_sizes(self):
    cols, rows = (self._cols, self._rows)
    cols_sh, rows_sh = (self._cols_sh, self._rows_sh)
    cols_sh_min, rows_sh_min = (self._cols_sh_min, self._rows_sh_min)
    cols_sh_max, rows_sh_max = (self._cols_sh_max, self._rows_sh_max)
    self._cols_count = cols_count = [defaultdict(int) for _ in cols]
    self._rows_count = rows_count = [defaultdict(int) for _ in rows]
    idx_iter = self._create_idx_iter(len(cols), len(rows))
    has_bound_y = has_bound_x = False
    for opt, (col, row) in zip(self.view_opts, idx_iter):
        (shw, shh), (w, h) = (opt['size_hint'], opt['size'])
        shw_min, shh_min = opt['size_hint_min']
        shw_max, shh_max = opt['size_hint_max']
        if shw is None:
            cols_count[col][w] += 1
        if shh is None:
            rows_count[row][h] += 1
        if shw is None:
            cols[col] = nmax(cols[col], w)
        else:
            cols_sh[col] = nmax(cols_sh[col], shw)
            if shw_min is not None:
                has_bound_x = True
                cols_sh_min[col] = nmax(cols_sh_min[col], shw_min)
            if shw_max is not None:
                has_bound_x = True
                cols_sh_max[col] = nmin(cols_sh_max[col], shw_max)
        if shh is None:
            rows[row] = nmax(rows[row], h)
        else:
            rows_sh[row] = nmax(rows_sh[row], shh)
            if shh_min is not None:
                has_bound_y = True
                rows_sh_min[row] = nmax(rows_sh_min[row], shh_min)
            if shh_max is not None:
                has_bound_y = True
                rows_sh_max[row] = nmin(rows_sh_max[row], shh_max)
    self._has_hint_bound_x = has_bound_x
    self._has_hint_bound_y = has_bound_y