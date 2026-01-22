from collections import deque
from . import errors, revision
def _find_gdfo(self):
    nodes = self._nodes
    known_parent_gdfos = {}
    pending = []
    for node in self._find_tails():
        node.gdfo = 1
        pending.append(node)
    while pending:
        node = pending.pop()
        for child_key in node.child_keys:
            child = nodes[child_key]
            if child_key in known_parent_gdfos:
                known_gdfo = known_parent_gdfos[child_key] + 1
                present = True
            else:
                known_gdfo = 1
                present = False
            if child.gdfo is None or node.gdfo + 1 > child.gdfo:
                child.gdfo = node.gdfo + 1
            if known_gdfo == len(child.parent_keys):
                pending.append(child)
                if present:
                    del known_parent_gdfos[child_key]
            else:
                known_parent_gdfos[child_key] = known_gdfo