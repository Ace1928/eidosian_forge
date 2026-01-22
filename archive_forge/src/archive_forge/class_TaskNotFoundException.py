from boto.exception import JSONResponseError
class TaskNotFoundException(JSONResponseError):
    pass