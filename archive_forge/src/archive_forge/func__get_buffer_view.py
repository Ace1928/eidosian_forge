def _get_buffer_view(in_obj):
    if isinstance(in_obj, str):
        raise TypeError('Unicode-objects must be encoded before calculating a CRC')
    mv = memoryview(in_obj)
    if mv.ndim > 1:
        raise BufferError('Buffer must be single dimension')
    return mv