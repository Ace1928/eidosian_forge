from boto.exception import JSONResponseError
class InvalidSubscriptionState(JSONResponseError):
    pass