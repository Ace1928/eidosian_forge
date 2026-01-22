def _crossing_to_edge(K, G, c):
    """Returns the edge of G corresponding to crossing c, where G is either the black graph or white graph."""
    edge = list()
    verts = G.vertices()
    black_faces = list()
    for v in verts:
        black_faces.append([x[0] for x in v])
    for i in range(len(black_faces)):
        if c in black_faces[i]:
            edge.append(verts[i])
    edge.append(c)
    if len(edge) != 3:
        raise Exception('Did not find two faces incident to c=' + repr(c))
    return tuple(edge)