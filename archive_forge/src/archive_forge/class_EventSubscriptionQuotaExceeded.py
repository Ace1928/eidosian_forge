from boto.exception import JSONResponseError
class EventSubscriptionQuotaExceeded(JSONResponseError):
    pass