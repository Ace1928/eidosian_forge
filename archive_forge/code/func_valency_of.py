from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def valency_of(self, obj):
    """Return the valency (degree) of a vertex in the graph."""
    try:
        l = [len(self[obj])]
    except KeyError:
        return 0
    for node in self[obj]:
        l.append(self.valency_of(node))
    return sum(l)