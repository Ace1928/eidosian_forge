import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
@open_file(1, mode='wb')
def write_graphml_xml(G, path, encoding='utf-8', prettyprint=True, infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None):
    """Write G in GraphML XML format to path

    Parameters
    ----------
    G : graph
       A networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    infer_numeric_types : boolean
       Determine if numeric types should be generalized.
       For example, if edges have both int and float 'weight' attributes,
       we infer in GraphML that both are floats.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_graphml(G, "test.graphml")

    Notes
    -----
    This implementation does not support mixed graphs (directed
    and unidirected edges together) hyperedges, nested graphs, or ports.
    """
    writer = GraphMLWriter(encoding=encoding, prettyprint=prettyprint, infer_numeric_types=infer_numeric_types, named_key_ids=named_key_ids, edge_id_from_attribute=edge_id_from_attribute)
    writer.add_graph_element(G)
    writer.dump(path)