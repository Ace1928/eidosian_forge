import functools
import itertools
from collections.abc import Iterable, Sequence
import numpy as np
from pennylane.pytrees import register_pytree
@staticmethod
def all_wires(list_of_wires, sort=False):
    """Return the wires that appear in any of the Wires objects in the list.

        This is similar to a set combine method, but keeps the order of wires as they appear in the list.

        Args:
            list_of_wires (List[Wires]): List of Wires objects
            sort (bool): Toggle for sorting the combined wire labels. The sorting is based on
                value if all keys are int, else labels' str representations are used.

        Returns:
            Wires: combined wires

        **Example**

        >>> wires1 = Wires([4, 0, 1])
        >>> wires2 = Wires([3, 0, 4])
        >>> wires3 = Wires([5, 3])
        >>> list_of_wires = [wires1, wires2, wires3]
        >>> Wires.all_wires(list_of_wires)
        <Wires = [4, 0, 1, 3, 5]>
        """
    converted_wires = (wires if isinstance(wires, Wires) else Wires(wires) for wires in list_of_wires)
    all_wires_list = itertools.chain(*(w.labels for w in converted_wires))
    combined = list(dict.fromkeys(all_wires_list))
    if sort:
        if all((isinstance(w, int) for w in combined)):
            combined = sorted(combined)
        else:
            combined = sorted(combined, key=str)
    return Wires(tuple(combined), _override=True)