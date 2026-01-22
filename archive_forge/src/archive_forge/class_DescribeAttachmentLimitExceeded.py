from boto.exception import JSONResponseError
class DescribeAttachmentLimitExceeded(JSONResponseError):
    pass