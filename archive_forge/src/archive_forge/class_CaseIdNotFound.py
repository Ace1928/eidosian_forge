from boto.exception import JSONResponseError
class CaseIdNotFound(JSONResponseError):
    pass