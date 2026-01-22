import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _ensure_write_partitioning(part, schema, flavor):
    if isinstance(part, PartitioningFactory):
        raise ValueError('A PartitioningFactory cannot be used. Did you call the partitioning function without supplying a schema?')
    if isinstance(part, Partitioning) and flavor:
        raise ValueError('Providing a partitioning_flavor with a Partitioning object is not supported')
    elif isinstance(part, (tuple, list)):
        part = partitioning(schema=pa.schema([schema.field(f) for f in part]), flavor=flavor)
    elif part is None:
        part = partitioning(pa.schema([]), flavor=flavor)
    if not isinstance(part, Partitioning):
        raise ValueError('partitioning must be a Partitioning object or a list of column names')
    return part