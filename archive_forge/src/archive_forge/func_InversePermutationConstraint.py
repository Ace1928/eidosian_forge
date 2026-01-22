from sys import version_info as _swig_python_version_info
import weakref
def InversePermutationConstraint(self, left, right):
    """
        Creates a constraint that enforces that 'left' and 'right' both
        represent permutations of [0..left.size()-1], and that 'right' is
        the inverse permutation of 'left', i.e. for all i in
        [0..left.size()-1], right[left[i]] = i.
        """
    return _pywrapcp.Solver_InversePermutationConstraint(self, left, right)