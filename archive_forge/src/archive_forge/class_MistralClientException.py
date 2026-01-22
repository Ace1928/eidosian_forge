class MistralClientException(Exception):
    """Base Exception for Mistral client

    To correctly use this class, inherit from it and define
    a 'message' and 'code' properties.
    """
    message = 'An unknown exception occurred'
    code = 'UNKNOWN_EXCEPTION'

    def __str__(self):
        return self.message

    def __init__(self, message=message):
        self.message = message
        super(MistralClientException, self).__init__('%s: %s' % (self.code, self.message))