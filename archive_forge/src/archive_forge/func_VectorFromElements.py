import array
import contextlib
import enum
import struct
@InMap
def VectorFromElements(self, elements):
    """Encodes sequence of any elements as a vector.

    Args:
      elements: sequence of elements, they may have different types.
    """
    with self.Vector():
        for e in elements:
            self.Add(e)