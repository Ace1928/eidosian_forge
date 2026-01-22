from sage.all import QQ, vector, matrix, VectorSpace
class Vector4(Vector):

    def __init__(self, entries):
        assert len(entries) == 4
        Vector.__init__(self, entries)