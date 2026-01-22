from boto.exception import JSONResponseError
class ClusterSecurityGroupAlreadyExistsFault(JSONResponseError):
    pass