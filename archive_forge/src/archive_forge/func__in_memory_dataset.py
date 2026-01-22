import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _in_memory_dataset(source, schema=None, **kwargs):
    if any((v is not None for v in kwargs.values())):
        raise ValueError('For in-memory datasets, you cannot pass any additional arguments')
    return InMemoryDataset(source, schema)