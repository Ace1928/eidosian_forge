from boto.exception import BotoServerError
class SESAddressBlacklistedError(SESError):
    """
    After you attempt to send mail to an address, and delivery repeatedly
    fails, said address is blacklisted for at least 24 hours. The blacklisting
    eventually expires, and you are able to attempt delivery again. If you
    attempt to send mail to a blacklisted email, this is raised.
    """
    pass