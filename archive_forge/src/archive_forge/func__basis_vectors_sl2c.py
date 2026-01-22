from ..matrix import matrix
def _basis_vectors_sl2c(CF):
    return [matrix([[1, 0], [0, 1]], ring=CF), matrix([[1, 0], [0, -1]], ring=CF), matrix([[0, 1], [1, 0]], ring=CF), matrix([[0, 1j], [-1j, 0]], ring=CF)]