import botocore.exceptions
class RetriesExceededError(Boto3Error):

    def __init__(self, last_exception, msg='Max Retries Exceeded'):
        super(RetriesExceededError, self).__init__(msg)
        self.last_exception = last_exception