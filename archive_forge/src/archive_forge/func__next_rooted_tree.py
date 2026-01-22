import networkx as nx
def _next_rooted_tree(predecessor, p=None):
    """One iteration of the Beyer-Hedetniemi algorithm."""
    if p is None:
        p = len(predecessor) - 1
        while predecessor[p] == 1:
            p -= 1
    if p == 0:
        return None
    q = p - 1
    while predecessor[q] != predecessor[p] - 1:
        q -= 1
    result = list(predecessor)
    for i in range(p, len(result)):
        result[i] = result[i - p + q]
    return result