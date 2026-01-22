from collections import namedtuple
from cvxpy.problems.problem import Problem
from cvxpy.utilities.deterministic import unique_list
class XpressProblem(Problem):
    """A convex optimization problem associated with the Xpress Optimizer

    Attributes
    ----------
    objective : Minimize or Maximize
        The expression to minimize or maximize.
    constraints : list
        The constraints on the problem variables.
    """
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, objective, constraints=None) -> None:
        super(XpressProblem, self).__init__(objective, constraints)
        self._iis = None

    def _reset_iis(self) -> None:
        """Clears the iis information
        """
        self._iis = None
        self._transferRow = None

    def __repr__(self) -> str:
        return 'XpressProblem(%s, %s)' % (repr(self.objective), repr(self.constraints))

    def __neg__(self) -> 'XpressProblem':
        return XpressProblem(-self.objective, self.constraints)

    def __add__(self, other):
        if other == 0:
            return self
        elif not isinstance(other, XpressProblem):
            raise NotImplementedError()
        return XpressProblem(self.objective + other.objective, unique_list(self.constraints + other.constraints))

    def __sub__(self, other):
        if not isinstance(other, XpressProblem):
            raise NotImplementedError()
        return XpressProblem(self.objective - other.objective, unique_list(self.constraints + other.constraints))

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return XpressProblem(self.objective * other, self.constraints)

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return XpressProblem(self.objective * (1.0 / other), self.constraints)