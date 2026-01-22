from boto.exception import JSONResponseError
class AuthorizationQuotaExceededFault(JSONResponseError):
    pass