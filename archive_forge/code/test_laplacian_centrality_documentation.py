import pytest
import networkx as nx
laplacian_centrality on a unconnected node graph should return 0

    For graphs without edges, the Laplacian energy is 0 and is unchanged with
    node removal, so::

        LC(v) = LE(G) - LE(G - v) = 0 - 0 = 0
    