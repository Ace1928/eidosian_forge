import pyarrow as pa
def _from_jvm_int_type(jvm_type):
    """
    Convert a JVM int type to its Python equivalent.

    Parameters
    ----------
    jvm_type : org.apache.arrow.vector.types.pojo.ArrowType$Int

    Returns
    -------
    typ : pyarrow.DataType
    """
    bit_width = jvm_type.getBitWidth()
    if jvm_type.getIsSigned():
        if bit_width == 8:
            return pa.int8()
        elif bit_width == 16:
            return pa.int16()
        elif bit_width == 32:
            return pa.int32()
        elif bit_width == 64:
            return pa.int64()
    elif bit_width == 8:
        return pa.uint8()
    elif bit_width == 16:
        return pa.uint16()
    elif bit_width == 32:
        return pa.uint32()
    elif bit_width == 64:
        return pa.uint64()