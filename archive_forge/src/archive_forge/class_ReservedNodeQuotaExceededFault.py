from boto.exception import JSONResponseError
class ReservedNodeQuotaExceededFault(JSONResponseError):
    pass