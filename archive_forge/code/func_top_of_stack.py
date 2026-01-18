from collections import defaultdict
import networkx as nx
def top_of_stack(l):
    """Returns the element on top of the stack."""
    if not l:
        return None
    return l[-1]