import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('fap_only_kwarg', ({'arrowstyle': '-'}, {'arrowsize': 20}, {'connectionstyle': 'arc3,rad=0.2'}, {'min_source_margin': 10}, {'min_target_margin': 10}))
def test_user_warnings_for_unused_edge_drawing_kwargs(fap_only_kwarg):
    """Users should get a warning when they specify a non-default value for
    one of the kwargs that applies only to edges drawn with FancyArrowPatches,
    but FancyArrowPatches aren't being used under the hood."""
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G}
    fig, ax = plt.subplots()
    kwarg_name = list(fap_only_kwarg.keys())[0]
    with pytest.warns(UserWarning, match=f'\n\nThe {kwarg_name} keyword argument is not applicable'):
        nx.draw_networkx_edges(G, pos, ax=ax, **fap_only_kwarg)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, **fap_only_kwarg)
    plt.delaxes(ax)