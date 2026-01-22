from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.backupdr import util
class ManagementServersClient(util.BackupDrClientBase):
    """Cloud Backup and DR Management client."""

    def __init__(self):
        super(ManagementServersClient, self).__init__()
        self.service = self.client.projects_locations_managementServers

    def Create(self, resource, network):
        networks = [self.messages.NetworkConfig(network=network)]
        parent = resource.Parent().RelativeName()
        management_server_id = resource.Name()
        management_server = self.messages.ManagementServer(networks=networks, type=self.messages.ManagementServer.TypeValueValuesEnum.BACKUP_RESTORE)
        request = self.messages.BackupdrProjectsLocationsManagementServersCreateRequest(parent=parent, managementServer=management_server, managementServerId=management_server_id)
        return self.service.Create(request)

    def Delete(self, resource):
        request = self.messages.BackupdrProjectsLocationsManagementServersDeleteRequest(name=resource.RelativeName())
        return self.service.Delete(request)