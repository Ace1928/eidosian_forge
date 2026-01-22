from boto.exception import JSONResponseError
class HsmConfigurationAlreadyExists(JSONResponseError):
    pass