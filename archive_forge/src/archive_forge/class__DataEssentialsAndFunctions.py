from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
class _DataEssentialsAndFunctions:

    def __init__(self, G, multigraph, demand='demand', capacity='capacity', weight='weight'):
        self.node_list = list(G)
        self.node_indices = {u: i for i, u in enumerate(self.node_list)}
        self.node_demands = [G.nodes[u].get(demand, 0) for u in self.node_list]
        self.edge_sources = []
        self.edge_targets = []
        if multigraph:
            self.edge_keys = []
        self.edge_indices = {}
        self.edge_capacities = []
        self.edge_weights = []
        if not multigraph:
            edges = G.edges(data=True)
        else:
            edges = G.edges(data=True, keys=True)
        inf = float('inf')
        edges = (e for e in edges if e[0] != e[1] and e[-1].get(capacity, inf) != 0)
        for i, e in enumerate(edges):
            self.edge_sources.append(self.node_indices[e[0]])
            self.edge_targets.append(self.node_indices[e[1]])
            if multigraph:
                self.edge_keys.append(e[2])
            self.edge_indices[e[:-1]] = i
            self.edge_capacities.append(e[-1].get(capacity, inf))
            self.edge_weights.append(e[-1].get(weight, 0))
        self.edge_count = None
        self.edge_flow = None
        self.node_potentials = None
        self.parent = None
        self.parent_edge = None
        self.subtree_size = None
        self.next_node_dft = None
        self.prev_node_dft = None
        self.last_descendent_dft = None
        self._spanning_tree_initialized = False

    def initialize_spanning_tree(self, n, faux_inf):
        self.edge_count = len(self.edge_indices)
        self.edge_flow = list(chain(repeat(0, self.edge_count), (abs(d) for d in self.node_demands)))
        self.node_potentials = [faux_inf if d <= 0 else -faux_inf for d in self.node_demands]
        self.parent = list(chain(repeat(-1, n), [None]))
        self.parent_edge = list(range(self.edge_count, self.edge_count + n))
        self.subtree_size = list(chain(repeat(1, n), [n + 1]))
        self.next_node_dft = list(chain(range(1, n), [-1, 0]))
        self.prev_node_dft = list(range(-1, n))
        self.last_descendent_dft = list(chain(range(n), [n - 1]))
        self._spanning_tree_initialized = True

    def find_apex(self, p, q):
        """
        Find the lowest common ancestor of nodes p and q in the spanning tree.
        """
        size_p = self.subtree_size[p]
        size_q = self.subtree_size[q]
        while True:
            while size_p < size_q:
                p = self.parent[p]
                size_p = self.subtree_size[p]
            while size_p > size_q:
                q = self.parent[q]
                size_q = self.subtree_size[q]
            if size_p == size_q:
                if p != q:
                    p = self.parent[p]
                    size_p = self.subtree_size[p]
                    q = self.parent[q]
                    size_q = self.subtree_size[q]
                else:
                    return p

    def trace_path(self, p, w):
        """
        Returns the nodes and edges on the path from node p to its ancestor w.
        """
        Wn = [p]
        We = []
        while p != w:
            We.append(self.parent_edge[p])
            p = self.parent[p]
            Wn.append(p)
        return (Wn, We)

    def find_cycle(self, i, p, q):
        """
        Returns the nodes and edges on the cycle containing edge i == (p, q)
        when the latter is added to the spanning tree.

        The cycle is oriented in the direction from p to q.
        """
        w = self.find_apex(p, q)
        Wn, We = self.trace_path(p, w)
        Wn.reverse()
        We.reverse()
        if We != [i]:
            We.append(i)
        WnR, WeR = self.trace_path(q, w)
        del WnR[-1]
        Wn += WnR
        We += WeR
        return (Wn, We)

    def augment_flow(self, Wn, We, f):
        """
        Augment f units of flow along a cycle represented by Wn and We.
        """
        for i, p in zip(We, Wn):
            if self.edge_sources[i] == p:
                self.edge_flow[i] += f
            else:
                self.edge_flow[i] -= f

    def trace_subtree(self, p):
        """
        Yield the nodes in the subtree rooted at a node p.
        """
        yield p
        l = self.last_descendent_dft[p]
        while p != l:
            p = self.next_node_dft[p]
            yield p

    def remove_edge(self, s, t):
        """
        Remove an edge (s, t) where parent[t] == s from the spanning tree.
        """
        size_t = self.subtree_size[t]
        prev_t = self.prev_node_dft[t]
        last_t = self.last_descendent_dft[t]
        next_last_t = self.next_node_dft[last_t]
        self.parent[t] = None
        self.parent_edge[t] = None
        self.next_node_dft[prev_t] = next_last_t
        self.prev_node_dft[next_last_t] = prev_t
        self.next_node_dft[last_t] = t
        self.prev_node_dft[t] = last_t
        while s is not None:
            self.subtree_size[s] -= size_t
            if self.last_descendent_dft[s] == last_t:
                self.last_descendent_dft[s] = prev_t
            s = self.parent[s]

    def make_root(self, q):
        """
        Make a node q the root of its containing subtree.
        """
        ancestors = []
        while q is not None:
            ancestors.append(q)
            q = self.parent[q]
        ancestors.reverse()
        for p, q in zip(ancestors, islice(ancestors, 1, None)):
            size_p = self.subtree_size[p]
            last_p = self.last_descendent_dft[p]
            prev_q = self.prev_node_dft[q]
            last_q = self.last_descendent_dft[q]
            next_last_q = self.next_node_dft[last_q]
            self.parent[p] = q
            self.parent[q] = None
            self.parent_edge[p] = self.parent_edge[q]
            self.parent_edge[q] = None
            self.subtree_size[p] = size_p - self.subtree_size[q]
            self.subtree_size[q] = size_p
            self.next_node_dft[prev_q] = next_last_q
            self.prev_node_dft[next_last_q] = prev_q
            self.next_node_dft[last_q] = q
            self.prev_node_dft[q] = last_q
            if last_p == last_q:
                self.last_descendent_dft[p] = prev_q
                last_p = prev_q
            self.prev_node_dft[p] = last_q
            self.next_node_dft[last_q] = p
            self.next_node_dft[last_p] = q
            self.prev_node_dft[q] = last_p
            self.last_descendent_dft[q] = last_p

    def add_edge(self, i, p, q):
        """
        Add an edge (p, q) to the spanning tree where q is the root of a subtree.
        """
        last_p = self.last_descendent_dft[p]
        next_last_p = self.next_node_dft[last_p]
        size_q = self.subtree_size[q]
        last_q = self.last_descendent_dft[q]
        self.parent[q] = p
        self.parent_edge[q] = i
        self.next_node_dft[last_p] = q
        self.prev_node_dft[q] = last_p
        self.prev_node_dft[next_last_p] = last_q
        self.next_node_dft[last_q] = next_last_p
        while p is not None:
            self.subtree_size[p] += size_q
            if self.last_descendent_dft[p] == last_p:
                self.last_descendent_dft[p] = last_q
            p = self.parent[p]

    def update_potentials(self, i, p, q):
        """
        Update the potentials of the nodes in the subtree rooted at a node
        q connected to its parent p by an edge i.
        """
        if q == self.edge_targets[i]:
            d = self.node_potentials[p] - self.edge_weights[i] - self.node_potentials[q]
        else:
            d = self.node_potentials[p] + self.edge_weights[i] - self.node_potentials[q]
        for q in self.trace_subtree(q):
            self.node_potentials[q] += d

    def reduced_cost(self, i):
        """Returns the reduced cost of an edge i."""
        c = self.edge_weights[i] - self.node_potentials[self.edge_sources[i]] + self.node_potentials[self.edge_targets[i]]
        return c if self.edge_flow[i] == 0 else -c

    def find_entering_edges(self):
        """Yield entering edges until none can be found."""
        if self.edge_count == 0:
            return
        B = int(ceil(sqrt(self.edge_count)))
        M = (self.edge_count + B - 1) // B
        m = 0
        f = 0
        while m < M:
            l = f + B
            if l <= self.edge_count:
                edges = range(f, l)
            else:
                l -= self.edge_count
                edges = chain(range(f, self.edge_count), range(l))
            f = l
            i = min(edges, key=self.reduced_cost)
            c = self.reduced_cost(i)
            if c >= 0:
                m += 1
            else:
                if self.edge_flow[i] == 0:
                    p = self.edge_sources[i]
                    q = self.edge_targets[i]
                else:
                    p = self.edge_targets[i]
                    q = self.edge_sources[i]
                yield (i, p, q)
                m = 0

    def residual_capacity(self, i, p):
        """Returns the residual capacity of an edge i in the direction away
        from its endpoint p.
        """
        return self.edge_capacities[i] - self.edge_flow[i] if self.edge_sources[i] == p else self.edge_flow[i]

    def find_leaving_edge(self, Wn, We):
        """Returns the leaving edge in a cycle represented by Wn and We."""
        j, s = min(zip(reversed(We), reversed(Wn)), key=lambda i_p: self.residual_capacity(*i_p))
        t = self.edge_targets[j] if self.edge_sources[j] == s else self.edge_sources[j]
        return (j, s, t)