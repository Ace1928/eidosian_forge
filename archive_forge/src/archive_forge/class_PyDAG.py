import sys
import functools
from .rustworkx import *
import rustworkx.visit
class PyDAG(PyDiGraph):
    """A class for creating direct acyclic graphs.

    PyDAG is just an alias of the PyDiGraph class and behaves identically to
    the :class:`~rustworkx.PyDiGraph` class and can be used interchangably
    with ``PyDiGraph``. It currently exists solely as a backwards
    compatibility alias for users of rustworkx from prior to the
    0.4.0 release when there was no PyDiGraph class.

    The PyDAG class is used to create a directed graph. It can be a
    multigraph (have multiple edges between nodes). Each node and edge
    (although rarely used for edges) is indexed by an integer id. These ids
    are stable for the lifetime of the graph object and on node or edge
    deletions you can have holes in the list of indices for the graph.
    Node indices will be reused on additions after removal. For example:

    .. jupyter-execute::

        import rustworkx as rx

        graph = rx.PyDAG()
        graph.add_nodes_from(list(range(5)))
        graph.add_nodes_from(list(range(2)))
        graph.remove_node(2)
        print("After deletion:", graph.node_indices())
        res_manual = graph.add_parent(6, None, None)
        print("After adding a new node:", graph.node_indices())

    Additionally, each node and edge contains an arbitrary Python object as a
    weight/data payload.

    You can use the index for access to the data payload as in the
    following example:

    .. jupyter-execute::

        import rustworkx as rx

        graph = rx.PyDAG()
        data_payload = "An arbitrary Python object"
        node_index = graph.add_node(data_payload)
        print("Node Index: %s" % node_index)
        print(graph[node_index])

    The PyDAG class implements the Python mapping protocol for nodes so in
    addition to access you can also update the data payload with:

    .. jupyter-execute::

        import rustworkx as rx

        graph = rx.PyDAG()
        data_payload = "An arbitrary Python object"
        node_index = graph.add_node(data_payload)
        graph[node_index] = "New Payload"
        print("Node Index: %s" % node_index)
        print(graph[node_index])

    The PyDAG class has an option for real time cycle checking which can
    be used to ensure any edges added to the graph does not introduce a cycle.
    By default the real time cycle checking feature is disabled for
    performance, however you can enable it by setting the ``check_cycle``
    attribute to True. For example::

        import rustworkx as rx
        dag = rx.PyDAG()
        dag.check_cycle = True

    or at object creation::

        import rustworkx as rx
        dag = rx.PyDAG(check_cycle=True)

    With check_cycle set to true any calls to :meth:`PyDAG.add_edge` will
    ensure that no cycles are added, ensuring that the PyDAG class truly
    represents a directed acyclic graph. Do note that this cycle checking on
    :meth:`~PyDAG.add_edge`, :meth:`~PyDigraph.add_edges_from`,
    :meth:`~PyDAG.add_edges_from_no_data`,
    :meth:`~PyDAG.extend_from_edge_list`,  and
    :meth:`~PyDAG.extend_from_weighted_edge_list` comes with a performance
    penalty that grows as the graph does.  If you're adding a node and edge at
    the same time, leveraging :meth:`PyDAG.add_child` or
    :meth:`PyDAG.add_parent` will avoid this overhead.

    By default a ``PyDAG`` is a multigraph (meaning there can be parallel
    edges between nodes) however this can be disabled by setting the
    ``multigraph`` kwarg to ``False`` when calling the ``PyDAG`` constructor.
    For example::

        import rustworkx as rx
        dag = rx.PyDAG(multigraph=False)

    This can only be set at ``PyDiGraph`` initialization and not adjusted after
    creation. When :attr:`~rustworkx.PyDiGraph.multigraph` is set to ``False``
    if a method call is made that would add a parallel edge it will instead
    update the existing edge's weight/data payload.

    The maximum number of nodes and edges allowed on a ``PyGraph`` object is
    :math:`2^{32} - 1` (4,294,967,294) each. Attempting to add more nodes or
    edges than this will result in an exception being raised.

    :param bool check_cycle: When this is set to ``True`` the created
        ``PyDAG`` has runtime cycle detection enabled.
    :param bool multgraph: When this is set to ``False`` the created
        ``PyDAG`` object will not be a multigraph. When ``False`` if a method
        call is made that would add parallel edges the the weight/weight from
        that method call will be used to update the existing edge in place.
    """
    pass