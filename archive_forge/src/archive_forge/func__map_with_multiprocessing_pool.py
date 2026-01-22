import contextlib
from multiprocessing import Pool, RLock
from tqdm.auto import tqdm
from ..utils import experimental, logging
def _map_with_multiprocessing_pool(function, iterable, num_proc, types, disable_tqdm, desc, single_map_nested_func):
    num_proc = num_proc if num_proc <= len(iterable) else len(iterable)
    split_kwds = []
    for index in range(num_proc):
        div = len(iterable) // num_proc
        mod = len(iterable) % num_proc
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        split_kwds.append((function, iterable[start:end], types, index, disable_tqdm, desc))
    if len(iterable) != sum((len(i[1]) for i in split_kwds)):
        raise ValueError(f'Error dividing inputs iterable among processes. Total number of objects {len(iterable)}, length: {sum((len(i[1]) for i in split_kwds))}')
    logger.info(f'Spawning {num_proc} processes for {len(iterable)} objects in slices of {[len(i[1]) for i in split_kwds]}')
    initargs, initializer = (None, None)
    if not disable_tqdm:
        initargs, initializer = ((RLock(),), tqdm.set_lock)
    with Pool(num_proc, initargs=initargs, initializer=initializer) as pool:
        mapped = pool.map(single_map_nested_func, split_kwds)
    logger.info(f'Finished {num_proc} processes')
    mapped = [obj for proc_res in mapped for obj in proc_res]
    logger.info(f'Unpacked {len(mapped)} objects')
    return mapped