from boto.exception import BotoServerError
class RefundAmountExceeded(ResponseError):
    """The refund amount is more than the refundable amount.
    """