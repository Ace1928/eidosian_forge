import numpy as np
import heapq
def _revalidate_node_edges(rag, node, heap_list):
    """Handles validation and invalidation of edges incident to a node.

    This function invalidates all existing edges incident on `node` and inserts
    new items in `heap_list` updated with the valid weights.

    rag : RAG
        The Region Adjacency Graph.
    node : int
        The id of the node whose incident edges are to be validated/invalidated
        .
    heap_list : list
        The list containing the existing heap of edges.
    """
    for nbr in rag.neighbors(node):
        data = rag[node][nbr]
        try:
            data['heap item'][3] = False
            _invalidate_edge(rag, node, nbr)
        except KeyError:
            pass
        wt = data['weight']
        heap_item = [wt, node, nbr, True]
        data['heap item'] = heap_item
        heapq.heappush(heap_list, heap_item)