from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
class ConfigReporter(object):
    """A container class to hold config report fields and methods."""

    def __init__(self, service):
        self.client = services_util.GetClientInstance()
        self.messages = services_util.GetMessagesModule()
        self.service = service
        self.old_config = ConfigReporterValue(service)
        self.new_config = ConfigReporterValue(service)

    def ConstructRequestMessage(self):
        old_config_value = self.old_config.ConstructConfigValue(self.messages.GenerateConfigReportRequest.OldConfigValue)
        new_config_value = self.new_config.ConstructConfigValue(self.messages.GenerateConfigReportRequest.NewConfigValue)
        return self.messages.GenerateConfigReportRequest(oldConfig=old_config_value, newConfig=new_config_value)

    def RunReport(self):
        result = self.client.services.GenerateConfigReport(self.ConstructRequestMessage())
        if not result:
            return None
        if not result.changeReports:
            return []
        return result.changeReports[0]