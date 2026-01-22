class ClusterCrossSlotError(ResponseError):
    """
    Error indicated CROSSSLOT error received from cluster.
    A CROSSSLOT error is generated when keys in a request don't hash to the
    same slot.
    """
    message = "Keys in request don't hash to the same slot"