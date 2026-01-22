from datetime import datetime
from boto.compat import six
class DeleteEnvironmentConfigurationResponse(Response):

    def __init__(self, response):
        response = response['DeleteEnvironmentConfigurationResponse']
        super(DeleteEnvironmentConfigurationResponse, self).__init__(response)