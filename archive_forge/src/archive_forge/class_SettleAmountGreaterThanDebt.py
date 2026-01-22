from boto.exception import BotoServerError
class SettleAmountGreaterThanDebt(ResponseError):
    """The amount being settled or written off is
       greater than the current debt.
    """