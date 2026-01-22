from boto.exception import BotoServerError
class UnverifiedAccount_Recipient(ResponseError):
    """The recipient's account must have a verified bank account or a
       credit card before this transaction can be initiated.
    """