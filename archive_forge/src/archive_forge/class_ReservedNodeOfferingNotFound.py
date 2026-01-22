from boto.exception import JSONResponseError
class ReservedNodeOfferingNotFound(JSONResponseError):
    pass