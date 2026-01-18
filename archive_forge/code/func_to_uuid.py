from __future__ import absolute_import
import uuid
from typing import Any, Iterable
def to_uuid(*args: Any) -> str:
    """Determine the uuid by input arguments. It will search the input recursively.
    If an object contains `__uuid__` method, it will call that method to get the uuid
    for that object.

    .. admonition:: Examples

        .. code-block:: python

            to_uuid([1,2,3])
            to_uuid(1,2,3)
            to_uuid(dict(a=1,b="z"))

    :param args: arbitrary input

    :return: uuid string
    """
    s = str(uuid.uuid5(uuid.NAMESPACE_DNS, ''))
    for a in args:
        for x in _get_strs(a):
            s = str(uuid.uuid5(uuid.NAMESPACE_DNS, s + x))
    return s