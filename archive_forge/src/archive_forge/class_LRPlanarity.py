from collections import defaultdict
import networkx as nx
class LRPlanarity:
    """A class to maintain the state during planarity check."""
    __slots__ = ['G', 'roots', 'height', 'lowpt', 'lowpt2', 'nesting_depth', 'parent_edge', 'DG', 'adjs', 'ordered_adjs', 'ref', 'side', 'S', 'stack_bottom', 'lowpt_edge', 'left_ref', 'right_ref', 'embedding']

    def __init__(self, G):
        self.G = nx.Graph()
        self.G.add_nodes_from(G.nodes)
        for e in G.edges:
            if e[0] != e[1]:
                self.G.add_edge(e[0], e[1])
        self.roots = []
        self.height = defaultdict(lambda: None)
        self.lowpt = {}
        self.lowpt2 = {}
        self.nesting_depth = {}
        self.parent_edge = defaultdict(lambda: None)
        self.DG = nx.DiGraph()
        self.DG.add_nodes_from(G.nodes)
        self.adjs = {}
        self.ordered_adjs = {}
        self.ref = defaultdict(lambda: None)
        self.side = defaultdict(lambda: 1)
        self.S = []
        self.stack_bottom = {}
        self.lowpt_edge = {}
        self.left_ref = {}
        self.right_ref = {}
        self.embedding = PlanarEmbedding()

    def lr_planarity(self):
        """Execute the LR planarity test.

        Returns
        -------
        embedding : dict
            If the graph is planar an embedding is returned. Otherwise None.
        """
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            return None
        for v in self.G:
            self.adjs[v] = list(self.G[v])
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation(v)
        self.G = None
        self.lowpt2 = None
        self.adjs = None
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
        for v in self.roots:
            if not self.dfs_testing(v):
                return None
        self.height = None
        self.lowpt = None
        self.S = None
        self.stack_bottom = None
        self.lowpt_edge = None
        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign(e) * self.nesting_depth[e]
        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge_cw(v, w, previous_node)
                previous_node = w
        self.DG = None
        self.nesting_depth = None
        self.ref = None
        for v in self.roots:
            self.dfs_embedding(v)
        self.roots = None
        self.parent_edge = None
        self.ordered_adjs = None
        self.left_ref = None
        self.right_ref = None
        self.side = None
        return self.embedding

    def lr_planarity_recursive(self):
        """Recursive version of :meth:`lr_planarity`."""
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            return None
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation_recursive(v)
        self.G = None
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
        for v in self.roots:
            if not self.dfs_testing_recursive(v):
                return None
        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign_recursive(e) * self.nesting_depth[e]
        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge_cw(v, w, previous_node)
                previous_node = w
        for v in self.roots:
            self.dfs_embedding_recursive(v)
        return self.embedding

    def dfs_orientation(self, v):
        """Orient the graph by DFS, compute lowpoints and nesting order."""
        dfs_stack = [v]
        ind = defaultdict(lambda: 0)
        skip_init = defaultdict(lambda: False)
        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]
            for w in self.adjs[v][ind[v]:]:
                vw = (v, w)
                if not skip_init[vw]:
                    if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                        ind[v] += 1
                        continue
                    self.DG.add_edge(v, w)
                    self.lowpt[vw] = self.height[v]
                    self.lowpt2[vw] = self.height[v]
                    if self.height[w] is None:
                        self.parent_edge[w] = vw
                        self.height[w] = self.height[v] + 1
                        dfs_stack.append(v)
                        dfs_stack.append(w)
                        skip_init[vw] = True
                        break
                    else:
                        self.lowpt[vw] = self.height[w]
                self.nesting_depth[vw] = 2 * self.lowpt[vw]
                if self.lowpt2[vw] < self.height[v]:
                    self.nesting_depth[vw] += 1
                if e is not None:
                    if self.lowpt[vw] < self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                        self.lowpt[e] = self.lowpt[vw]
                    elif self.lowpt[vw] > self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                    else:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])
                ind[v] += 1

    def dfs_orientation_recursive(self, v):
        """Recursive version of :meth:`dfs_orientation`."""
        e = self.parent_edge[v]
        for w in self.G[v]:
            if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                continue
            vw = (v, w)
            self.DG.add_edge(v, w)
            self.lowpt[vw] = self.height[v]
            self.lowpt2[vw] = self.height[v]
            if self.height[w] is None:
                self.parent_edge[w] = vw
                self.height[w] = self.height[v] + 1
                self.dfs_orientation_recursive(w)
            else:
                self.lowpt[vw] = self.height[w]
            self.nesting_depth[vw] = 2 * self.lowpt[vw]
            if self.lowpt2[vw] < self.height[v]:
                self.nesting_depth[vw] += 1
            if e is not None:
                if self.lowpt[vw] < self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                    self.lowpt[e] = self.lowpt[vw]
                elif self.lowpt[vw] > self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                else:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])

    def dfs_testing(self, v):
        """Test for LR partition."""
        dfs_stack = [v]
        ind = defaultdict(lambda: 0)
        skip_init = defaultdict(lambda: False)
        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]
            skip_final = False
            for w in self.ordered_adjs[v][ind[v]:]:
                ei = (v, w)
                if not skip_init[ei]:
                    self.stack_bottom[ei] = top_of_stack(self.S)
                    if ei == self.parent_edge[w]:
                        dfs_stack.append(v)
                        dfs_stack.append(w)
                        skip_init[ei] = True
                        skip_final = True
                        break
                    else:
                        self.lowpt_edge[ei] = ei
                        self.S.append(ConflictPair(right=Interval(ei, ei)))
                if self.lowpt[ei] < self.height[v]:
                    if w == self.ordered_adjs[v][0]:
                        self.lowpt_edge[e] = self.lowpt_edge[ei]
                    elif not self.add_constraints(ei, e):
                        return False
                ind[v] += 1
            if not skip_final:
                if e is not None:
                    self.remove_back_edges(e)
        return True

    def dfs_testing_recursive(self, v):
        """Recursive version of :meth:`dfs_testing`."""
        e = self.parent_edge[v]
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            self.stack_bottom[ei] = top_of_stack(self.S)
            if ei == self.parent_edge[w]:
                if not self.dfs_testing_recursive(w):
                    return False
            else:
                self.lowpt_edge[ei] = ei
                self.S.append(ConflictPair(right=Interval(ei, ei)))
            if self.lowpt[ei] < self.height[v]:
                if w == self.ordered_adjs[v][0]:
                    self.lowpt_edge[e] = self.lowpt_edge[ei]
                elif not self.add_constraints(ei, e):
                    return False
        if e is not None:
            self.remove_back_edges(e)
        return True

    def add_constraints(self, ei, e):
        P = ConflictPair()
        while True:
            Q = self.S.pop()
            if not Q.left.empty():
                Q.swap()
            if not Q.left.empty():
                return False
            if self.lowpt[Q.right.low] > self.lowpt[e]:
                if P.right.empty():
                    P.right = Q.right.copy()
                else:
                    self.ref[P.right.low] = Q.right.high
                P.right.low = Q.right.low
            else:
                self.ref[Q.right.low] = self.lowpt_edge[e]
            if top_of_stack(self.S) == self.stack_bottom[ei]:
                break
        while top_of_stack(self.S).left.conflicting(ei, self) or top_of_stack(self.S).right.conflicting(ei, self):
            Q = self.S.pop()
            if Q.right.conflicting(ei, self):
                Q.swap()
            if Q.right.conflicting(ei, self):
                return False
            self.ref[P.right.low] = Q.right.high
            if Q.right.low is not None:
                P.right.low = Q.right.low
            if P.left.empty():
                P.left = Q.left.copy()
            else:
                self.ref[P.left.low] = Q.left.high
            P.left.low = Q.left.low
        if not (P.left.empty() and P.right.empty()):
            self.S.append(P)
        return True

    def remove_back_edges(self, e):
        u = e[0]
        while self.S and top_of_stack(self.S).lowest(self) == self.height[u]:
            P = self.S.pop()
            if P.left.low is not None:
                self.side[P.left.low] = -1
        if self.S:
            P = self.S.pop()
            while P.left.high is not None and P.left.high[1] == u:
                P.left.high = self.ref[P.left.high]
            if P.left.high is None and P.left.low is not None:
                self.ref[P.left.low] = P.right.low
                self.side[P.left.low] = -1
                P.left.low = None
            while P.right.high is not None and P.right.high[1] == u:
                P.right.high = self.ref[P.right.high]
            if P.right.high is None and P.right.low is not None:
                self.ref[P.right.low] = P.left.low
                self.side[P.right.low] = -1
                P.right.low = None
            self.S.append(P)
        if self.lowpt[e] < self.height[u]:
            hl = top_of_stack(self.S).left.high
            hr = top_of_stack(self.S).right.high
            if hl is not None and (hr is None or self.lowpt[hl] > self.lowpt[hr]):
                self.ref[e] = hl
            else:
                self.ref[e] = hr

    def dfs_embedding(self, v):
        """Completes the embedding."""
        dfs_stack = [v]
        ind = defaultdict(lambda: 0)
        while dfs_stack:
            v = dfs_stack.pop()
            for w in self.ordered_adjs[v][ind[v]:]:
                ind[v] += 1
                ei = (v, w)
                if ei == self.parent_edge[w]:
                    self.embedding.add_half_edge_first(w, v)
                    self.left_ref[v] = w
                    self.right_ref[v] = w
                    dfs_stack.append(v)
                    dfs_stack.append(w)
                    break
                elif self.side[ei] == 1:
                    self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
                else:
                    self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
                    self.left_ref[w] = v

    def dfs_embedding_recursive(self, v):
        """Recursive version of :meth:`dfs_embedding`."""
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            if ei == self.parent_edge[w]:
                self.embedding.add_half_edge_first(w, v)
                self.left_ref[v] = w
                self.right_ref[v] = w
                self.dfs_embedding_recursive(w)
            elif self.side[ei] == 1:
                self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
            else:
                self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
                self.left_ref[w] = v

    def sign(self, e):
        """Resolve the relative side of an edge to the absolute side."""
        dfs_stack = [e]
        old_ref = defaultdict(lambda: None)
        while dfs_stack:
            e = dfs_stack.pop()
            if self.ref[e] is not None:
                dfs_stack.append(e)
                dfs_stack.append(self.ref[e])
                old_ref[e] = self.ref[e]
                self.ref[e] = None
            else:
                self.side[e] *= self.side[old_ref[e]]
        return self.side[e]

    def sign_recursive(self, e):
        """Recursive version of :meth:`sign`."""
        if self.ref[e] is not None:
            self.side[e] = self.side[e] * self.sign_recursive(self.ref[e])
            self.ref[e] = None
        return self.side[e]