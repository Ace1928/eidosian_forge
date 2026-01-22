import operator
import warnings
def _edit_dist_backtrace(lev):
    i, j = (len(lev) - 1, len(lev[0]) - 1)
    alignment = [(i, j)]
    while (i, j) != (0, 0):
        directions = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        direction_costs = ((lev[i][j] if i >= 0 and j >= 0 else float('inf'), (i, j)) for i, j in directions)
        _, (i, j) = min(direction_costs, key=operator.itemgetter(0))
        alignment.append((i, j))
    return list(reversed(alignment))