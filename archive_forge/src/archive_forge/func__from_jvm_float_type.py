import pyarrow as pa
def _from_jvm_float_type(jvm_type):
    """
    Convert a JVM float type to its Python equivalent.

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$FloatingPoint

    Returns
    -------
    typ: pyarrow.DataType
    """
    precision = jvm_type.getPrecision().toString()
    if precision == 'HALF':
        return pa.float16()
    elif precision == 'SINGLE':
        return pa.float32()
    elif precision == 'DOUBLE':
        return pa.float64()