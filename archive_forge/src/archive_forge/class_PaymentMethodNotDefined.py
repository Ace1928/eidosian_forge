from boto.exception import BotoServerError
class PaymentMethodNotDefined(ResponseError):
    """Payment method is not defined in the transaction.
    """