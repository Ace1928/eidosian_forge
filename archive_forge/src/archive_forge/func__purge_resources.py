import sys
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
def _purge_resources(self, neutron_client, resource_types, tenant_resources):
    deleted = {}
    failed = {}
    failures = False
    for resources in tenant_resources:
        index = tenant_resources.index(resources)
        resource_type = resource_types[index]
        failed[resource_type] = 0
        deleted[resource_type] = 0
        for resource in resources:
            try:
                self._delete_resource(neutron_client, resource_type, resource)
                deleted[resource_type] += 1
                self.deleted_resources += 1
            except Exception:
                failures = True
                failed[resource_type] += 1
                self.total_resources -= 1
            percent_complete = 100
            if self.total_resources > 0:
                percent_complete = self.deleted_resources / float(self.total_resources) * 100
            sys.stdout.write('\rPurging resources: %d%% complete.' % percent_complete)
            sys.stdout.flush()
    return (deleted, failed, failures)