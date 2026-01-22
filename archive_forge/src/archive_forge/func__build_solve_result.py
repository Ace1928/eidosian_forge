import json
from typing import List, Optional, Tuple
from google.protobuf import json_format
import requests
from ortools.service.v1 import optimization_pb2
from ortools.math_opt import rpc_pb2
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.ipc import proto_converter
def _build_solve_result(json_response: bytes, model: mathopt.Model) -> Tuple[mathopt.SolveResult, List[str]]:
    """Parses a JSON representation of a response to a SolveResult object.

    Args:
      json_response: bytes representing the `SolveMathOptModelResponse` in JSON
        format
      model: The optimization model that was solved

    Returns:
      A SolveResult of the model.
      A list of messages with the logs.

    Raises:
      SerializationError: If parsing the json response fails or if converting the
        OR API response to the internal MathOpt response fails.
    """
    try:
        api_response = json_format.Parse(json_response, optimization_pb2.SolveMathOptModelResponse())
    except json_format.ParseError as json_err:
        raise ValueError('API response is not a valid SolveMathOptModelResponse JSON') from json_err
    response = proto_converter.convert_response(api_response)
    return (mathopt.parse_solve_result(response.result, model), list(response.messages))