from boto.exception import JSONResponseError
class SNSTopicArnNotFound(JSONResponseError):
    pass