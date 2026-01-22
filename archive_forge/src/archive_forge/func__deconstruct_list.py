from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@classmethod
def _deconstruct_list(cls, lst: Iterable, output: List, stack: List, result_consumers: List['DeferredExecution'], out_append: Callable):
    """
        Deconstruct the specified list.

        Parameters
        ----------
        lst : list
        output : list
        stack : list
        result_consumers : list
        out_append : Callable
            The reference to the ``list.append()`` method.

        Yields
        ------
        Generator
            Either ``_deconstruct_list()`` or ``_deconstruct_chain()`` generator.
        """
    for obj in lst:
        if isinstance(obj, DeferredExecution):
            if (out_pos := getattr(obj, 'out_pos', None)):
                obj.unsubscribe()
                if obj.has_result:
                    out_append(obj.data)
                else:
                    out_append(_Tag.REF)
                    out_append(out_pos)
                    output[out_pos] = out_pos
                    if obj.subscribers == 0:
                        output[out_pos + 1] = 0
                        result_consumers.remove(obj)
            else:
                out_append(_Tag.CHAIN)
                yield cls._deconstruct_chain(obj, output, stack, result_consumers)
                out_append(_Tag.END)
        elif isinstance(obj, ListOrTuple):
            out_append(_Tag.LIST)
            yield cls._deconstruct_list(obj, output, stack, result_consumers, out_append)
        else:
            out_append(obj)
    out_append(_Tag.END)