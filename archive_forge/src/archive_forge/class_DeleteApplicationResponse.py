from datetime import datetime
from boto.compat import six
class DeleteApplicationResponse(Response):

    def __init__(self, response):
        response = response['DeleteApplicationResponse']
        super(DeleteApplicationResponse, self).__init__(response)