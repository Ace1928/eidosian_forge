import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def test_one(self, pred_dates, succ_dates):
    """
        Edges one hop out from Gx_node in the mapping should be
        time-respecting with respect to each other, regardless of
        direction.
        """
    time_respecting = True
    dates = pred_dates + succ_dates
    if any((x is None for x in dates)):
        raise ValueError('Date or datetime not supplied for at least one edge.')
    dates.sort()
    if 0 < len(dates) and (not dates[-1] - dates[0] <= self.delta):
        time_respecting = False
    return time_respecting