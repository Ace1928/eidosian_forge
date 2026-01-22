from datetime import datetime
from boto.compat import six
class ApplicationDescription(BaseObject):

    def __init__(self, response):
        super(ApplicationDescription, self).__init__()
        self.application_name = str(response['ApplicationName'])
        self.configuration_templates = []
        if response['ConfigurationTemplates']:
            for member in response['ConfigurationTemplates']:
                configuration_template = str(member)
                self.configuration_templates.append(configuration_template)
        self.date_created = datetime.fromtimestamp(response['DateCreated'])
        self.date_updated = datetime.fromtimestamp(response['DateUpdated'])
        self.description = str(response['Description'])
        self.versions = []
        if response['Versions']:
            for member in response['Versions']:
                version = str(member)
                self.versions.append(version)