from .cartan_type import CartanType
from sympy.core.basic import Atom
def add_simple_roots(self, root1, root2):
    """Add two simple roots together

        The function takes as input two integers, root1 and root2.  It then
        uses these integers as keys in the dictionary of simple roots, and gets
        the corresponding simple roots, and then adds them together.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> newroot = c.add_simple_roots(1, 2)
        >>> newroot
        [1, 0, -1, 0]

        """
    alpha = self.simple_roots()
    if root1 > len(alpha) or root2 > len(alpha):
        raise ValueError("You've used a root that doesn't exist!")
    a1 = alpha[root1]
    a2 = alpha[root2]
    newroot = [_a1 + _a2 for _a1, _a2 in zip(a1, a2)]
    return newroot