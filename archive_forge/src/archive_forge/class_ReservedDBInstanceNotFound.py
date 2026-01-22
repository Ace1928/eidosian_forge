from boto.exception import JSONResponseError
class ReservedDBInstanceNotFound(JSONResponseError):
    pass