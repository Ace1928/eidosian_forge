from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _ptolemy_equations_action_by_decoration_change(self, N):
    """
        This is reimplementing get_ptolemy_equations_action_by_decoration_change
        from addl_code/ptolemy_equations.c

        We can change a decoration by multiplying a coset of a cusp by a
        diagonal matrix. Let's let a diagonal matrix SL(n,C) with diagonal
        entries 1 1 ... z 1 ... 1 1/z (z at position j) act on cusp i. It
        changes some Ptolemy coordinate c_p_t by some power z^n.
        This is expressed in the following matrix as the entry in row
        labelled c_p_t and the column labelled diagonal_entry_j_on_cusp_i.
        """
    matrix = []
    row_explanations = []
    for tet_index, tet in enumerate(self.getTetrahedra()):
        for pt in utilities.quadruples_with_fixed_sum_iterator(N, skipVertices=True):
            row = (N - 1) * self.getNumberOfVertices() * [0]
            for vertex in range(4):
                cusp_index = self.vertexIndex(tet.getVertex(vertex))
                for diag_entry in range(pt[vertex]):
                    column_index = cusp_index * (N - 1) + diag_entry
                    row[column_index] += 1
            matrix.append(row)
            row_explanations.append('c_%d%d%d%d' % pt + '_%d' % tet_index)
    column_explanations = ['diagonal_entry_%d_on_cusp_%d' % (diag_entry, cusp_index) for cusp_index in range(self.getNumberOfVertices()) for diag_entry in range(N - 1)]
    return (matrix, row_explanations, column_explanations)