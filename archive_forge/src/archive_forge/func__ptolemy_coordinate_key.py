from ...snap import t3mlite as t3m
def _ptolemy_coordinate_key(tet_index, edge):
    return 'c_%d%d%d%d_%d' % ((edge & 8) >> 3, (edge & 4) >> 2, (edge & 2) >> 1, edge & 1, tet_index)