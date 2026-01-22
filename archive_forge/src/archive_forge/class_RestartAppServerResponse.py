from datetime import datetime
from boto.compat import six
class RestartAppServerResponse(Response):

    def __init__(self, response):
        response = response['RestartAppServerResponse']
        super(RestartAppServerResponse, self).__init__(response)