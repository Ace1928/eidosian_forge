class TryAgainError(ResponseError):
    """
    Error indicated TRYAGAIN error received from cluster.
    Operations on keys that don't exist or are - during resharding - split
    between the source and destination nodes, will generate a -TRYAGAIN error.
    """

    def __init__(self, *args, **kwargs):
        pass