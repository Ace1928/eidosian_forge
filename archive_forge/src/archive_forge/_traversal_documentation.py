import cupy
import cupyx.scipy.sparse
Analyzes the connected components of a sparse graph

    Args:
        csgraph (cupy.ndarray of cupyx.scipy.sparse.csr_matrix): The adjacency
            matrix representing connectivity among nodes.
        directed (bool): If ``True``, it operates on a directed graph. If
            ``False``, it operates on an undirected graph.
        connection (str): ``'weak'`` or ``'strong'``. For directed graphs, the
            type of connection to use. Nodes i and j are "strongly" connected
            only when a path exists both from i to j and from j to i.
            If ``directed`` is ``False``, this argument is ignored.
        return_labels (bool): If ``True``, it returns the labels for each of
            the connected components.

    Returns:
        tuple of int and cupy.ndarray, or int:
            If ``return_labels`` == ``True``, returns a tuple ``(n, labels)``,
            where ``n`` is the number of connected components and ``labels`` is
            labels of each connected components. Otherwise, returns ``n``.

    .. seealso:: :func:`scipy.sparse.csgraph.connected_components`
    