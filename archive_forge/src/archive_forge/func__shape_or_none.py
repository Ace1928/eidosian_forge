import cupy
def _shape_or_none(M):
    if M is not None:
        return M.shape
    else:
        return (None,) * 2