import pyarrow as pa
def _from_jvm_timestamp_type(jvm_type):
    """
    Convert a JVM timestamp type to its Python equivalent.

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$Timestamp

    Returns
    -------
    typ: pyarrow.DataType
    """
    time_unit = jvm_type.getUnit().toString()
    timezone = jvm_type.getTimezone()
    if timezone is not None:
        timezone = str(timezone)
    if time_unit == 'SECOND':
        return pa.timestamp('s', tz=timezone)
    elif time_unit == 'MILLISECOND':
        return pa.timestamp('ms', tz=timezone)
    elif time_unit == 'MICROSECOND':
        return pa.timestamp('us', tz=timezone)
    elif time_unit == 'NANOSECOND':
        return pa.timestamp('ns', tz=timezone)