from _pydevd_bundle.pydevd_constants import IS_PY311_OR_GREATER
import dis
from types import CodeType
from collections import namedtuple

    Calculate smart step into variants for the given line range.
    :param frame:
    :type frame: :py:class:`types.FrameType`
    :param start_line:
    :param end_line:
    :return: A list of call names from the first to the last.
    :note: it's guaranteed that the offsets appear in order.
    :raise: :py:class:`RuntimeError` if failed to parse the bytecode or if dis cannot be used.
    