import networkx as nx
from networkx.utils.decorators import not_implemented_for
def _plain_bfs(G, source):
    """A fast BFS node generator

    The direction of the edge between nodes is ignored.

    For directed graphs only.

    """
    n = len(G)
    Gsucc = G._succ
    Gpred = G._pred
    seen = {source}
    nextlevel = [source]
    yield source
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in Gsucc[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield w
            for w in Gpred[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield w
            if len(seen) == n:
                return