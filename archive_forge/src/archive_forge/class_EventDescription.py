from datetime import datetime
from boto.compat import six
class EventDescription(BaseObject):

    def __init__(self, response):
        super(EventDescription, self).__init__()
        self.application_name = str(response['ApplicationName'])
        self.environment_name = str(response['EnvironmentName'])
        self.event_date = datetime.fromtimestamp(response['EventDate'])
        self.message = str(response['Message'])
        self.request_id = str(response['RequestId'])
        self.severity = str(response['Severity'])
        self.template_name = str(response['TemplateName'])
        self.version_label = str(response['VersionLabel'])