from boto.exception import JSONResponseError
class StorageQuotaExceeded(JSONResponseError):
    pass