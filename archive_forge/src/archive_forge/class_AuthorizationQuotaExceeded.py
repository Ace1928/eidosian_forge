from boto.exception import JSONResponseError
class AuthorizationQuotaExceeded(JSONResponseError):
    pass