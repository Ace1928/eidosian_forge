from boto.exception import JSONResponseError
class InstanceQuotaExceeded(JSONResponseError):
    pass