from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
class ConfigReporterValue(object):
    """A container class to hold config report value fields and methods."""
    SERVICE_CONFIG_TYPE_URL = 'type.googleapis.com/google.api.Service'
    CONFIG_REF_TYPE_URL = 'type.googleapis.com/google.api.servicemanagement.v1.ConfigRef'
    CONFIG_SOURCE_TYPE_URL = 'type.googleapis.com/google.api.servicemanagement.v1.ConfigSource'

    def __init__(self, service):
        self.messages = services_util.GetMessagesModule()
        self.service = service
        self.config = None
        self.swagger_path = None
        self.swagger_contents = None
        self.config_id = None
        self.config_use_active_id = True

    def SetConfig(self, config):
        self.config = config
        self.swagger_path = None
        self.swagger_contents = None
        self.config_id = None
        self.config_use_active_id = False

    def SetSwagger(self, path, contents):
        self.config = None
        self.swagger_path = path
        self.swagger_contents = contents
        self.config_id = None
        self.config_use_active_id = False

    def SetConfigId(self, config_id):
        self.config = None
        self.swagger_path = None
        self.swagger_contents = None
        self.config_id = config_id
        self.config_use_active_id = False

    def SetConfigUseDefaultId(self):
        self.config = None
        self.swagger_path = None
        self.swagger_contents = None
        self.config_id = None
        self.config_use_active_id = True

    def GetTypeUrl(self):
        if self.config:
            return ConfigReporterValue.SERVICE_CONFIG_TYPE_URL
        elif self.swagger_path and self.swagger_contents:
            return ConfigReporterValue.CONFIG_SOURCE_TYPE_URL
        elif self.config_id or self.config_use_active_id:
            return ConfigReporterValue.CONFIG_REF_TYPE_URL

    def IsReadyForReport(self):
        return self.config is not None or self.swagger_path is not None or self.config_id is not None or self.config_use_active_id

    def ConstructConfigValue(self, value_type):
        """Make a value to insert into the GenerateConfigReport request.

    Args:
      value_type: The type to encode the message into. Generally, either
        OldConfigValue or NewConfigValue.

    Returns:
      The encoded config value object of type value_type.
    """
        result = {}
        if not self.IsReadyForReport():
            return None
        elif self.config:
            result.update(self.config)
        elif self.swagger_path:
            config_file = self.messages.ConfigFile(filePath=self.swagger_path, fileContents=self.swagger_contents, fileType=self.messages.ConfigFile.FileTypeValueValuesEnum.OPEN_API_YAML)
            config_source_message = self.messages.ConfigSource(files=[config_file])
            result.update(encoding.MessageToDict(config_source_message))
        else:
            if self.config_id:
                resource = 'services/{0}/configs/{1}'.format(self.service, self.config_id)
            else:
                active_config_ids = services_util.GetActiveServiceConfigIdsForService(self.service)
                if active_config_ids:
                    resource = 'services/{0}/configs/{1}'.format(self.service, active_config_ids[0])
                else:
                    resource = 'services/{0}'.format(self.service)
            result.update({'name': resource})
        result.update({'@type': self.GetTypeUrl()})
        return encoding.DictToMessage(result, value_type)