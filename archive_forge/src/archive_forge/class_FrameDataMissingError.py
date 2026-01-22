import h2.errors
class FrameDataMissingError(ProtocolError):
    """
    The frame that we received is missing some data.

    .. versionadded:: 2.0.0
    """
    error_code = h2.errors.ErrorCodes.FRAME_SIZE_ERROR