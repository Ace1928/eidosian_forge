from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _create_labels(rects, horizontal, ax, rotation):
    """find the position of the label for each value of each category

    right now it supports only up to the four categories

    ax: the axis on which the label should be applied
    rotation: the rotation list for each side
    """
    categories = _categories_level(list(rects.keys()))
    if len(categories) > 4:
        msg = 'maximum of 4 level supported for axes labeling... and 4is already a lot of levels, are you sure you need them all?'
        raise ValueError(msg)
    labels = {}
    items = list(rects.items())
    vertical = not horizontal
    ax2 = ax.twinx()
    ax3 = ax.twiny()
    ticks_pos = [ax.set_xticks, ax.set_yticks, ax3.set_xticks, ax2.set_yticks]
    ticks_lab = [ax.set_xticklabels, ax.set_yticklabels, ax3.set_xticklabels, ax2.set_yticklabels]
    if vertical:
        ticks_pos = ticks_pos[1:] + ticks_pos[:1]
        ticks_lab = ticks_lab[1:] + ticks_lab[:1]
    for pos, lab in zip(ticks_pos, ticks_lab):
        pos([])
        lab([])
    for level_idx, level in enumerate(categories):
        level_ticks = dict()
        for value in level:
            if horizontal:
                if level_idx == 3:
                    index_select = [-1, -1, -1]
                else:
                    index_select = [+0, -1, -1]
            elif level_idx == 3:
                index_select = [+0, -1, +0]
            else:
                index_select = [-1, -1, -1]
            basekey = tuple((categories[i][index_select[i]] for i in range(level_idx)))
            basekey = basekey + (value,)
            subset = {k: v for k, v in items if basekey == k[:level_idx + 1]}
            vals = list(subset.values())
            W = sum((w * h for x, y, w, h in vals))
            x_lab = sum((_get_position(x, w, h, W) for x, y, w, h in vals))
            y_lab = sum((_get_position(y, h, w, W) for x, y, w, h in vals))
            side = (level_idx + vertical) % 4
            level_ticks[value] = y_lab if side % 2 else x_lab
        ticks_pos[level_idx](list(level_ticks.values()))
        ticks_lab[level_idx](list(level_ticks.keys()), rotation=rotation[level_idx])
    return labels