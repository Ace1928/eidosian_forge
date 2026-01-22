from datetime import datetime
from boto.compat import six
class ConfigurationOptionSetting(BaseObject):

    def __init__(self, response):
        super(ConfigurationOptionSetting, self).__init__()
        self.namespace = str(response['Namespace'])
        self.option_name = str(response['OptionName'])
        self.value = str(response['Value'])