from numba.tests.support import TestCase
def tests_numba_types(self):
    import numba.types
    import numba.core.types as types
    self.assertIsNot(numba.types, types)
    self.assertIs(numba.types.intp, types.intp)
    self.assertIs(numba.types.float64, types.float64)
    self.assertIs(numba.types.Array, types.Array)
    import numba.types.misc
    self.assertIs(types.misc, numba.types.misc)
    self.assertIs(types.misc.Optional, numba.types.misc.Optional)
    self.assertIs(types.StringLiteral, numba.types.misc.StringLiteral)
    from numba.types import containers
    self.assertIs(types.containers, containers)
    self.assertIs(types.containers.Sequence, containers.Sequence)
    from numba.types.containers import Sequence
    self.assertIs(Sequence, containers.Sequence)