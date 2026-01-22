import copy
from google.protobuf import message
from ortools.math_opt.python import normalize
class MathOptProtoEquivMatcher:
    """Matcher that checks if protos are equivalent in the MathOpt sense.

    See normalize.py for technical definitions.
    """

    def __init__(self, expected: message.Message):
        self._expected = copy.deepcopy(expected)
        normalize.math_opt_normalize_proto(self._expected)

    def __eq__(self, actual: message.Message) -> bool:
        actual = copy.deepcopy(actual)
        normalize.math_opt_normalize_proto(actual)
        return str(actual) == str(self._expected)

    def __ne__(self, other: message.Message) -> bool:
        return not self == other