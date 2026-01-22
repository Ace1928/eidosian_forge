from ...snap import t3mlite as t3m

    Given a cusp cross section, compute lifted Ptolemy coordinates
    (i.e., logarithms of the Ptolemy coordinates) returned as a dictionary
    (e.g., the key for the Ptolemy coordinate for the edge from
    vertex 0 to vertex 3 or simplex 4 is c_1001_4).

    For complete cusp cross sections (where no lifted_one_cocycle is
    necessary), we use Zickert's algorithm (Christian Zickert, The
    volume and Chern-Simons invariant of a representation, Duke
    Math. J. 150 no. 3 (2009) 489-532, math.GT/0710.2049). In this
    case, all values for keys corresponding to the same edge in the
    triangulation are guaranteed to be the same.

    For the incomplete cusp cross sections, a lifted_one_cocycle
    needs to be given. This cocycle is a lift of the cocycle one_cocycle
    given to ComplexCuspCrossSection.fromManifoldAndShapes.
    More precisely, lifted_one_cocycle is in C^1(boundary M; C) and
    needs to map to one_cocycle in C^1(boundary M; C^*).
    