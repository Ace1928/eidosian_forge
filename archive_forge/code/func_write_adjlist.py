import networkx as nx
from networkx.utils import open_file
@open_file(1, mode='wb')
def write_adjlist(G, path, comments='#', delimiter=' ', encoding='utf-8'):
    """Write graph G in single-line adjacency-list format to path.


    Parameters
    ----------
    G : NetworkX graph

    path : string or file
       Filename or file handle for data output.
       Filenames ending in .gz or .bz2 will be compressed.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels

    encoding : string, optional
       Text encoding.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_adjlist(G, "test.adjlist")

    The path can be a filehandle or a string with the name of the file. If a
    filehandle is provided, it has to be opened in 'wb' mode.

    >>> fh = open("test.adjlist", "wb")
    >>> nx.write_adjlist(G, fh)

    Notes
    -----
    The default `delimiter=" "` will result in unexpected results if node names contain
    whitespace characters. To avoid this problem, specify an alternate delimiter when spaces are
    valid in node names.
    NB: This option is not available for data that isn't user-generated.

    This format does not store graph, node, or edge data.

    See Also
    --------
    read_adjlist, generate_adjlist
    """
    import sys
    import time
    pargs = comments + ' '.join(sys.argv) + '\n'
    header = pargs + comments + f' GMT {time.asctime(time.gmtime())}\n' + comments + f' {G.name}\n'
    path.write(header.encode(encoding))
    for line in generate_adjlist(G, delimiter):
        line += '\n'
        path.write(line.encode(encoding))