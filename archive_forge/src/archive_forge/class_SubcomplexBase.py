from snappy.snap import t3mlite as t3m
class SubcomplexBase:

    def __init__(self, subcomplex_type, tet_and_perm):
        self.subcomplex_type = subcomplex_type
        self.tet_and_perm = tet_and_perm