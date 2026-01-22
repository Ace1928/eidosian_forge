import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
class TestAtlas:

    @classmethod
    def setup_class(cls):
        global atlas
        from networkx.generators import atlas
        cls.GAG = atlas.graph_atlas_g()

    def test_graph_atlas(self):
        Atlas = self.GAG[0:100]
        alphabet = list(range(26))
        for graph in Atlas:
            nlist = list(graph)
            labels = alphabet[:len(nlist)]
            for s in range(10):
                random.shuffle(labels)
                d = dict(zip(nlist, labels))
                relabel = nx.relabel_nodes(graph, d)
                gm = iso.GraphMatcher(graph, relabel)
                assert gm.is_isomorphic()