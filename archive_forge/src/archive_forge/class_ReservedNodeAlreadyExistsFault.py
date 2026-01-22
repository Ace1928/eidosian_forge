from boto.exception import JSONResponseError
class ReservedNodeAlreadyExistsFault(JSONResponseError):
    pass