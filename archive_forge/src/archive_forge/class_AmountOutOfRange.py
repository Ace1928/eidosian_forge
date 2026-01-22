from boto.exception import BotoServerError
class AmountOutOfRange(ResponseError):
    """The transaction amount is more than the allowed range.
    """