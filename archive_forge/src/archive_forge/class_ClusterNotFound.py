from boto.exception import JSONResponseError
class ClusterNotFound(JSONResponseError):
    pass