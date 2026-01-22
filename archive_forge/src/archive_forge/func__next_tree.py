import networkx as nx
def _next_tree(candidate):
    """One iteration of the Wright, Richmond, Odlyzko and McKay
    algorithm."""
    left, rest = _split_tree(candidate)
    left_height = max(left)
    rest_height = max(rest)
    valid = rest_height >= left_height
    if valid and rest_height == left_height:
        if len(left) > len(rest):
            valid = False
        elif len(left) == len(rest) and left > rest:
            valid = False
    if valid:
        return candidate
    else:
        p = len(left)
        new_candidate = _next_rooted_tree(candidate, p)
        if candidate[p] > 2:
            new_left, new_rest = _split_tree(new_candidate)
            new_left_height = max(new_left)
            suffix = range(1, new_left_height + 2)
            new_candidate[-len(suffix):] = suffix
        return new_candidate