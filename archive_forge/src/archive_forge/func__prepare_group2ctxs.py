import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def _prepare_group2ctxs(group2ctxs, ctx_len):
    """Prepare the group2contexts, will duplicate the context
    if some ctx_group map to only one context.
    """
    if group2ctxs is None:
        return [None] * ctx_len
    elif isinstance(group2ctxs, list):
        assert len(group2ctxs) == ctx_len, 'length of group2ctxs            should be %d' % ctx_len
        return group2ctxs
    elif isinstance(group2ctxs, dict):
        ret = [{} for i in range(ctx_len)]
        for k, v in group2ctxs.items():
            ctxs = None
            if isinstance(v, ctx.Context):
                ctxs = [v] * ctx_len
            elif len(v) == 1:
                ctxs = v * ctx_len
            else:
                assert len(v) == ctx_len, 'length of group2ctxs[%s]                        should be %d or 1' % (k, ctx_len)
                ctxs = v
            for i in range(ctx_len):
                ret[i][k] = ctxs[i]
        return ret
    else:
        assert False, 'group2ctxs should be list of dict of str to context,            or dict of str to context or list of context'
        return False