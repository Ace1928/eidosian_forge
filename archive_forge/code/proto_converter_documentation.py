from google.protobuf import message
from ortools.service.v1 import optimization_pb2
from ortools.math_opt import rpc_pb2
from ortools.math_opt.python import normalize
Converts a `SolveMathOptModelResponse` to a `SolveResponse`.

    Args:
      api_response: A `SolveMathOptModelResponse` response built from a MathOpt
        model.

    Returns:
      A `SolveResponse` response built from a MathOpt model.
    