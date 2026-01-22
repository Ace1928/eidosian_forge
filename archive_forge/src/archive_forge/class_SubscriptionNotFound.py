from boto.exception import JSONResponseError
class SubscriptionNotFound(JSONResponseError):
    pass