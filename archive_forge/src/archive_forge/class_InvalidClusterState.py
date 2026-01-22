from boto.exception import JSONResponseError
class InvalidClusterState(JSONResponseError):
    pass