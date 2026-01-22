import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
def all_gather_list(data, max_size=16384):
    """
    Gather arbitrary data from all nodes into a list.

    Similar to `~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    :param data:
        data from the local worker to be gathered on other workers
    :param int max_size:
        maximum size of the data to be gathered across workers

    :returns:
        a list containing [data1, data2, ...] of all workers
    """
    if not is_distributed():
        return [data]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
    buffer = all_gather_list._buffer
    buffer.zero_()
    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256
    buffer_rank = buffer[rank * max_size:(rank + 1) * max_size]
    buffer_rank[0] = enc_size // 255
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size + 2] = torch.ByteTensor(list(enc))
    dist.all_reduce(buffer)
    result = []
    for i in range(world_size):
        out_buffer = buffer[i * max_size:(i + 1) * max_size]
        size = 255 * out_buffer[0].item() + out_buffer[1].item()
        if size > 0:
            try:
                result.append(pickle.loads(bytes(out_buffer[2:size + 2].tolist())))
            except pickle.UnpicklingError:
                raise RuntimeError('There was an unpickling error in all_gather_list. This likely means your workers got out of syncronization (e.g. one is expecting to sync and another is not.)')
    return result