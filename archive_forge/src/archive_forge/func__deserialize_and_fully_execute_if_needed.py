import contextlib
import traceback
import ray
def _deserialize_and_fully_execute_if_needed(serialized_ds: bytes):
    ds = ray.data.Dataset.deserialize_lineage(serialized_ds)
    return ds