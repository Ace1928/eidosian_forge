import dataclasses
import datetime
import enum
from typing import Dict, Optional
from ortools.pdlp import solvers_pb2 as pdlp_solvers_pb2
from ortools.glop import parameters_pb2 as glop_parameters_pb2
from ortools.gscip import gscip_pb2
from ortools.math_opt import parameters_pb2 as math_opt_parameters_pb2
from ortools.math_opt.solvers import glpk_pb2
from ortools.math_opt.solvers import gurobi_pb2
from ortools.math_opt.solvers import highs_pb2
from ortools.math_opt.solvers import osqp_pb2
from ortools.sat import sat_parameters_pb2
@dataclasses.dataclass
class GurobiParameters:
    """Gurobi specific parameters for solving.

    See https://www.gurobi.com/documentation/9.1/refman/parameters.html for a list
    of possible parameters.

    Example use:
      gurobi=GurobiParameters();
      gurobi.param_values["BarIterLimit"] = "10";

    With Gurobi, the order that parameters are applied can have an impact in rare
    situations. Parameters are applied in the following order:
     * LogToConsole is set from SolveParameters.enable_output.
     * Any common parameters not overwritten by GurobiParameters.
     * param_values in iteration order (insertion order).
    We set LogToConsole first because setting other parameters can generate
    output.
    """
    param_values: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_proto(self) -> gurobi_pb2.GurobiParametersProto:
        return gurobi_pb2.GurobiParametersProto(parameters=[gurobi_pb2.GurobiParametersProto.Parameter(name=key, value=val) for key, val in self.param_values.items()])