import copy, logging
from pyomo.common.dependencies import numpy
def adj_lists(self, G, excludeEdges=None, nodes=None, multi=False):
    """
        Returns an adjacency list and a reverse adjacency list
        of node indexes for a MultiDiGraph.

        Arguments
        ---------
            G
                A networkx MultiDiGraph
            excludeEdges
                List of edge indexes to ignore when considering neighbors
            nodes
                List of nodes to form the adjacencies from
            multi
                If True, adjacency lists will contains tuples of
                (node, key) for every edge between two nodes

        Returns
        -------
            i2n
                Map from index to node for all nodes included in nodes
            adj
                Adjacency list of successor indexes
            adjR
                Reverse adjacency list of predecessor indexes
        """
    adj = []
    adjR = []
    exclude = set()
    if excludeEdges is not None:
        edge_list = self.idx_to_edge(G)
        for ei in excludeEdges:
            exclude.add(edge_list[ei])
    if nodes is None:
        nodes = self.idx_to_node(G)
    i2n = [None] * len(nodes)
    n2i = dict()
    i = -1
    for node in nodes:
        i += 1
        n2i[node] = i
        i2n[i] = node
    i = -1
    for node in nodes:
        i += 1
        adj.append([])
        adjR.append([])
        seen = set()
        for edge in G.out_edges(node, keys=True):
            suc, key = (edge[1], edge[2])
            if not multi and suc in seen:
                continue
            if suc in nodes and edge not in exclude:
                seen.add(suc)
                if multi:
                    adj[i].append((n2i[suc], key))
                else:
                    adj[i].append(n2i[suc])
        seen = set()
        for edge in G.in_edges(node, keys=True):
            pre, key = (edge[0], edge[2])
            if not multi and pre in seen:
                continue
            if pre in nodes and edge not in exclude:
                seen.add(pre)
                if multi:
                    adjR[i].append((n2i[pre], key))
                else:
                    adjR[i].append(n2i[pre])
    return (i2n, adj, adjR)