from datetime import datetime
from boto.compat import six
class SwapEnvironmentCNAMEsResponse(Response):

    def __init__(self, response):
        response = response['SwapEnvironmentCNAMEsResponse']
        super(SwapEnvironmentCNAMEsResponse, self).__init__(response)