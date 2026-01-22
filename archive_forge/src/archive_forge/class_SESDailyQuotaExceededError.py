from boto.exception import BotoServerError
class SESDailyQuotaExceededError(SESError):
    """
    Your account's daily (rolling 24 hour total) allotment of outbound emails
    has been exceeded.
    """
    pass