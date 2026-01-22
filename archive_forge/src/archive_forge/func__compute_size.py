from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, OptionProperty, \
def _compute_size(c, available_size, idx):
    sh_min = c.size_hint_min[idx]
    sh_max = c.size_hint_max[idx]
    val = c.size_hint[idx] * available_size
    if sh_min is not None:
        if sh_max is not None:
            return max(min(sh_max, val), sh_min)
        return max(val, sh_min)
    if sh_max is not None:
        return min(sh_max, val)
    return val