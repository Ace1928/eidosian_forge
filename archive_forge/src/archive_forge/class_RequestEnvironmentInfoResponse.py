from datetime import datetime
from boto.compat import six
class RequestEnvironmentInfoResponse(Response):

    def __init__(self, response):
        response = response['RequestEnvironmentInfoResponse']
        super(RequestEnvironmentInfoResponse, self).__init__(response)