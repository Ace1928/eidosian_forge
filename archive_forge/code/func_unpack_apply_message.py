import pickle
import warnings
from itertools import chain
from jupyter_client.session import MAX_BYTES, MAX_ITEMS
def unpack_apply_message(bufs, g=None, copy=True):
    """unpack f,args,kwargs from buffers packed by pack_apply_message()
    Returns: original f,args,kwargs"""
    bufs = list(bufs)
    assert len(bufs) >= 2, 'not enough buffers!'
    pf = bufs.pop(0)
    f = uncan(pickle.loads(pf), g)
    pinfo = bufs.pop(0)
    info = pickle.loads(pinfo)
    arg_bufs, kwarg_bufs = (bufs[:info['narg_bufs']], bufs[info['narg_bufs']:])
    args_list = []
    for _ in range(info['nargs']):
        arg, arg_bufs = deserialize_object(arg_bufs, g)
        args_list.append(arg)
    args = tuple(args_list)
    assert not arg_bufs, "Shouldn't be any arg bufs left over"
    kwargs = {}
    for key in info['kw_keys']:
        kwarg, kwarg_bufs = deserialize_object(kwarg_bufs, g)
        kwargs[key] = kwarg
    assert not kwarg_bufs, "Shouldn't be any kwarg bufs left over"
    return (f, args, kwargs)