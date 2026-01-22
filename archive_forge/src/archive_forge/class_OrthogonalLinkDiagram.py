import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
class OrthogonalLinkDiagram(list):
    """
    A link diagram where all edges are made up of horizontal and vertical
    segments.
    """

    def __init__(self, link):
        self.link = link = link.copy()
        list.__init__(self, [Face(link, F) for F in link.faces()])
        F = max(self, key=len)
        F.exterior = True
        self.face_network = self.flow_networkx()
        self.bend()
        self.orient_edges()
        self.edges = sum([F for F in self], [])
        self.repair_components()

    def flow_networkx(faces):
        """
        Tamassia's associated graph N(P) where the flow problem resides.
        """
        G = networkx.DiGraph()
        source_demand = sum((F.source_capacity() for F in faces))
        G.add_node('s', demand=-source_demand)
        for i, F in enumerate(faces):
            if F.source_capacity():
                G.add_edge('s', i, weight=0, capacity=F.source_capacity())
        sink_demand = sum((F.sink_capacity() for F in faces))
        assert sink_demand == source_demand
        G.add_node('t', demand=sink_demand)
        for i, F in enumerate(faces):
            if F.sink_capacity():
                G.add_edge(i, 't', weight=0, capacity=F.sink_capacity())
        for A in faces:
            for B in faces:
                if A != B and A.edge_of_intersection(B):
                    G.add_edge(faces.index(A), faces.index(B), weight=1)
        return G

    def bend(self):
        """
        Computes a minimal size set of edge bends that allows the link diagram
        to be embedded orthogonally. This follows directly Tamassia's first
        paper.
        """
        N = self.face_network
        flow = networkx.min_cost_flow(N)
        for a, flows in flow.items():
            for b, w_a in flows.items():
                if w_a and set(['s', 't']).isdisjoint(set([a, b])):
                    w_b = flow[b][a]
                    A, B = (self[a], self[b])
                    e_a, e_b = A.edge_of_intersection(B)
                    turns_a = w_a * [1] + w_b * [-1]
                    turns_b = w_b * [1] + w_a * [-1]
                    subdivide_edge(e_a, len(turns_a))
                    A.bend(e_a, turns_a)
                    B.bend(e_b, turns_b)

    def repair_components(self):
        """
        Repair the link components and store their numbering for
        easy reference.  Also order the strand_CEPs so they go component
        by component.
        """
        self.link.link_components = [component[0].component() for component in self.link.link_components]
        self.strand_CEP_to_component = stc = {}
        self.strand_CEPs = []
        for n, component in enumerate(self.link.link_components):
            for ce in component:
                if isinstance(ce.crossing, Strand):
                    stc[ce] = n
                    self.strand_CEPs.append(ce)

    def orient_edges(self):
        """
        For each edge in a face, assign it one of four possible orientations:
        "left", "right", "up", "down".
        """
        orientations = {self[0][0]: 'right'}
        G = self.face_network.to_undirected()
        (G.remove_node('s'), G.remove_node('t'))
        for i in networkx.dfs_preorder_nodes(G, 0):
            F = self[i]
            for edge in F:
                if edge in orientations:
                    new_orientations = F.orient_edges(edge, orientations[edge])
                    for e, dir in new_orientations.items():
                        if e in orientations:
                            assert orientations[e] == dir
                        else:
                            orientations[e] = dir
                    break
        assert len(orientations) == sum((len(F) for F in self))
        self.orientations = orientations

    def orthogonal_rep(self):
        orientations = self.orientations
        spec = [[(e.crossing, e.opposite().crossing) for e in self.edges if orientations[e] == dir] for dir in ['right', 'up']]
        return OrthogonalRep(*spec)

    def orthogonal_spec(self):
        orientations = self.orientations
        return [[(e.crossing.label, e.opposite().crossing.label) for e in self.edges if orientations[e] == dir] for dir in ['right', 'up']]

    def break_into_arrows(self):
        arrows = []
        for s in self.strand_CEPs:
            arrow = [s, s.next()]
            while not isinstance(arrow[-1].crossing, Strand):
                arrow.append(arrow[-1].next())
            arrows.append(arrow)
        undercrossings = {}
        for i, arrow in enumerate(arrows):
            for a in arrow[1:-1]:
                if a.is_under_crossing():
                    undercrossings[a] = i
        crossings = []
        for i, arrow in enumerate(arrows):
            for a in arrow[1:-1]:
                if a.is_over_crossing():
                    crossings.append((undercrossings[a.other()], i, False, a.crossing.label))
        return (arrows, crossings)

    def plink_data(self):
        """
        Returns:
        * a list of vertex positions
        * a list of arrows joining vertices
        * a list of crossings in the format (arrow over, arrow under)
        """
        emb = self.orthogonal_rep().basic_grid_embedding()
        x_max = max((a for a, b in emb.values()))
        y_max = max((b for a, b in emb.values()))
        vertex_positions = []
        for v in self.strand_CEPs:
            if x_max >= y_max:
                a, b = emb[v.crossing]
                b = y_max - b
            else:
                b, a = emb[v.crossing]
            vertex_positions.append((10 * (a + 1), 10 * (b + 1)))
        vert_indices = dict(((v, i) for i, v in enumerate(self.strand_CEPs)))
        arrows, crossings = self.break_into_arrows()
        arrows = [(vert_indices[a[0]], vert_indices[a[-1]]) for a in arrows]
        return (vertex_positions, arrows, crossings)