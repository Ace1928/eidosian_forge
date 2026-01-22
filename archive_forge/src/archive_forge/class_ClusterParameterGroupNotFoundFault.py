from boto.exception import JSONResponseError
class ClusterParameterGroupNotFoundFault(JSONResponseError):
    pass