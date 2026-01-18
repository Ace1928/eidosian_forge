import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def test_two(self, pred_dates, succ_dates):
    """
        Edges from a dual Gx_node in the mapping should be ordered in
        a time-respecting manner.
        """
    time_respecting = True
    pred_dates.sort()
    succ_dates.sort()
    if 0 < len(succ_dates) and 0 < len(pred_dates) and (succ_dates[0] < pred_dates[-1]):
        time_respecting = False
    return time_respecting