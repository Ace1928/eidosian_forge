from boto.exception import BotoServerError
class SESDomainEndsWithDotError(SESError):
    """
    Recipient's email address' domain ends with a period/dot.
    """
    pass