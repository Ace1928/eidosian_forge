from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
def _dual_cone(self, *args):
    """Implements the dual cone of the PSD cone See Pg 85 of the
        MOSEK modelling cookbook for more information"""
    if args is None:
        return self.dual_variables[0] >> 0
    else:
        args_shapes = [arg.shape for arg in args]
        instance_args_shapes = [arg.shape for arg in self.args]
        assert len(args) == len(self.args)
        assert args_shapes == instance_args_shapes
        return args[0] >> 0