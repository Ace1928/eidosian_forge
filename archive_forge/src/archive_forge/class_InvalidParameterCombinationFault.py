from boto.exception import JSONResponseError
class InvalidParameterCombinationFault(JSONResponseError):
    pass