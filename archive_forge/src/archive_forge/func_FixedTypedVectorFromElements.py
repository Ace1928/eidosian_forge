import array
import contextlib
import enum
import struct
@InMap
def FixedTypedVectorFromElements(self, elements, element_type=None, byte_width=0):
    """Encodes sequence of elements of the same type as fixed typed vector.

    Args:
      elements: Sequence of elements, they must be of the same type. Allowed
        types are `Type.INT`, `Type.UINT`, `Type.FLOAT`. Allowed number of
        elements are 2, 3, or 4.
      element_type: Suggested element type. Setting it to None means determining
        correct value automatically based on the given elements.
      byte_width: Number of bytes to use per element. For `Type.INT` and
        `Type.UINT`: 1, 2, 4, or 8. For `Type.FLOAT`: 4 or 8. Setting it to 0
        means determining correct value automatically based on the given
        elements.
    """
    if not 2 <= len(elements) <= 4:
        raise ValueError('only 2, 3, or 4 elements are supported')
    types = {type(e) for e in elements}
    if len(types) != 1:
        raise TypeError('all elements must be of the same type')
    type_, = types
    if element_type is None:
        element_type = {int: Type.INT, float: Type.FLOAT}.get(type_)
        if not element_type:
            raise TypeError('unsupported element_type: %s' % type_)
    if byte_width == 0:
        width = {Type.UINT: BitWidth.U, Type.INT: BitWidth.I, Type.FLOAT: BitWidth.F}[element_type]
        byte_width = 1 << max((width(e) for e in elements))
    self._WriteScalarVector(element_type, byte_width, elements, fixed=True)