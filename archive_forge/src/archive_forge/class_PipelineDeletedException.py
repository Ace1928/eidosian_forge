from boto.exception import JSONResponseError
class PipelineDeletedException(JSONResponseError):
    pass