from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.backupdr import util
class BackupPlanAssociationsClient(util.BackupDrClientBase):
    """Cloud Backup and DR Backup plan associations client."""

    def __init__(self):
        super(BackupPlanAssociationsClient, self).__init__()
        self.service = self.client.projects_locations_backupPlanAssociations

    def Create(self, bpa_resource, backup_plan, workload_resource):
        parent = bpa_resource.Parent().RelativeName()
        bpa_id = bpa_resource.Name()
        bpa = self.messages.BackupPlanAssociation(backupPlan=backup_plan.RelativeName(), resource=workload_resource)
        request = self.messages.BackupdrProjectsLocationsBackupPlanAssociationsCreateRequest(parent=parent, backupPlanAssociation=bpa, backupPlanAssociationId=bpa_id)
        return self.service.Create(request)

    def Delete(self, resource):
        request = self.messages.BackupdrProjectsLocationsBackupPlanAssociationsDeleteRequest(name=resource.RelativeName())
        return self.service.Delete(request)