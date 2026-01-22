from boto.exception import JSONResponseError
class SNSInvalidTopic(JSONResponseError):
    pass