from boto.exception import JSONResponseError
class PipelineNotFoundException(JSONResponseError):
    pass