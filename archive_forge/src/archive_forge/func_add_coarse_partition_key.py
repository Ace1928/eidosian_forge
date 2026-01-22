import pickle
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
from triad import Schema
from .._constants import _ZERO_COPY
def add_coarse_partition_key(df: rd.Dataset, keys: List[str], output_key: str, bucket: int) -> rd.Dataset:
    ray_remote_args: Dict[str, Any] = {'num_cpus': 1}

    def add_coarse_key(arrow_df: pa.Table) -> pa.Table:
        hdf = arrow_df.select(keys).to_pandas()
        _hash = pd.util.hash_pandas_object(hdf, index=False).mod(bucket)
        return arrow_df.append_column(output_key, pa.Array.from_pandas(_hash))
    return df.map_batches(add_coarse_key, batch_format='pyarrow', **_ZERO_COPY, **ray_remote_args)