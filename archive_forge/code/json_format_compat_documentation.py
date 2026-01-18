import math
Returns whether a protobuf Value will be serializable by MessageToJson.

    The json_format documentation states that "attempting to serialize NaN or
    Infinity results in error."

    https://protobuf.dev/reference/protobuf/google.protobuf/#value

    Args:
      value: A value of type protobuf.Value.

    Returns:
      True if the Value should be serializable without error by MessageToJson.
      False, otherwise.
    