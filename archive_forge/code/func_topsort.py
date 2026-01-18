from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def topsort(self):
    """Sort the graph topologically.

        Returns:
            List: of objects in the order in which they must be handled.
        """
    graph = DependencyGraph()
    components = self._tarjan72()
    NC = {node: component for component in components for node in component}
    for component in components:
        graph.add_arc(component)
    for node in self:
        node_c = NC[node]
        for successor in self[node]:
            successor_c = NC[successor]
            if node_c != successor_c:
                graph.add_edge(node_c, successor_c)
    return [t[0] for t in graph._khan62()]