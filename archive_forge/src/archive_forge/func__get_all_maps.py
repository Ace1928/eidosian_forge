from __future__ import absolute_import, print_function
from ..palette import Palette
def _get_all_maps():
    """
    Returns a dictionary of all Tableau palettes, including reversed ones.

    """
    d = dict(((name, get_map(name)) for name in palette_names))
    d.update(dict(((name + '_r', get_map(name, reverse=True)) for name in palette_names)))
    return d