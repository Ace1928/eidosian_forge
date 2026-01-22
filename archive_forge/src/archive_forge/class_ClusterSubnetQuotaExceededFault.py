from boto.exception import JSONResponseError
class ClusterSubnetQuotaExceededFault(JSONResponseError):
    pass