from boto.exception import JSONResponseError
class OptionGroupNotFound(JSONResponseError):
    pass