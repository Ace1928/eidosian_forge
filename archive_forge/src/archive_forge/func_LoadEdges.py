import collections
import json
import sys
def LoadEdges(filename, targets):
    """Load the edges map from the dump file, and filter it to only
  show targets in |targets| and their depedendents."""
    file = open('dump.json')
    edges = json.load(file)
    file.close()
    target_edges = {}
    to_visit = targets[:]
    while to_visit:
        src = to_visit.pop()
        if src in target_edges:
            continue
        target_edges[src] = edges[src]
        to_visit.extend(edges[src])
    return target_edges