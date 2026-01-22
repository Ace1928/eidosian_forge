from datetime import datetime
from boto.compat import six
class RetrieveEnvironmentInfoResponse(Response):

    def __init__(self, response):
        response = response['RetrieveEnvironmentInfoResponse']
        super(RetrieveEnvironmentInfoResponse, self).__init__(response)
        response = response['RetrieveEnvironmentInfoResult']
        self.environment_info = []
        if response['EnvironmentInfo']:
            for member in response['EnvironmentInfo']:
                environment_info = EnvironmentInfoDescription(member)
                self.environment_info.append(environment_info)