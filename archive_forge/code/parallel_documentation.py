import contextlib
from multiprocessing import Pool, RLock
from tqdm.auto import tqdm
from ..utils import experimental, logging

    **Experimental.**  Configures the parallel backend for parallelized dataset loading, which uses the parallelization
    implemented by joblib.

    Args:
        backend_name (str): Name of backend for parallelization implementation, has to be supported by joblib.

     Example usage:
     ```py
     with parallel_backend('spark'):
       dataset = load_dataset(..., num_proc=2)
     ```
    