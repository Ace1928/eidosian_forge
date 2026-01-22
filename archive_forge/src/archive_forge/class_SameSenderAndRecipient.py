from boto.exception import BotoServerError
class SameSenderAndRecipient(ResponseError):
    """The sender and receiver are identical, which is not allowed.
    """