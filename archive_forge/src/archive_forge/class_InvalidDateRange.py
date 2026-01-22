from boto.exception import BotoServerError
class InvalidDateRange(ResponseError):
    """The end date specified is before the start date or the start date
       is in the future.
    """