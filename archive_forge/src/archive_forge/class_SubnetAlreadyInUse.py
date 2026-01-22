from boto.exception import JSONResponseError
class SubnetAlreadyInUse(JSONResponseError):
    pass