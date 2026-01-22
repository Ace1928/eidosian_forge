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
@enum.unique
class Emphasis(enum.Enum):
    """Effort level applied to an optional task while solving (see SolveParameters for use).

    - OFF: disable this task.
    - LOW: apply reduced effort.
    - MEDIUM: typically the default setting (unless the default is off).
    - HIGH: apply extra effort beyond MEDIUM.
    - VERY_HIGH: apply the maximum effort.

    Typically used as Optional[Emphasis]. It used to configure a solver feature as
    follows:
      * If a solver doesn't support the feature, only None will always be valid,
        any other setting will give an invalid argument error (some solvers may
        also accept OFF).
      * If the solver supports the feature:
        - When set to None, the underlying default is used.
        - When the feature cannot be turned off, OFF will produce an error.
        - If the feature is enabled by default, the solver default is typically
          mapped to MEDIUM.
        - If the feature is supported, LOW, MEDIUM, HIGH, and VERY HIGH will never
          give an error, and will map onto their best match.

    This must stay synchronized with math_opt_parameters_pb2.EmphasisProto.
    """
    OFF = math_opt_parameters_pb2.EMPHASIS_OFF
    LOW = math_opt_parameters_pb2.EMPHASIS_LOW
    MEDIUM = math_opt_parameters_pb2.EMPHASIS_MEDIUM
    HIGH = math_opt_parameters_pb2.EMPHASIS_HIGH
    VERY_HIGH = math_opt_parameters_pb2.EMPHASIS_VERY_HIGH