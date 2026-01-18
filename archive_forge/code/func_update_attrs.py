import os
import tempfile
import networkx as nx
def update_attrs(which, attrs):
    added = []
    for k, v in attrs.items():
        if k not in G.graph[which]:
            G.graph[which][k] = v
            added.append(k)