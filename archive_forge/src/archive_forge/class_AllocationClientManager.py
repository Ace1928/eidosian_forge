from blazarclient import base
class AllocationClientManager(base.BaseClientManager):
    """Manager for the ComputeHost connected requests."""

    def get(self, resource, resource_id):
        """Get allocation for resource identified by type and ID."""
        resp, body = self.request_manager.get('/%s/%s/allocation' % (resource, resource_id))
        return body['allocation']

    def list(self, resource, sort_by=None):
        """List allocations for all resources of a type."""
        resp, body = self.request_manager.get('/%s/allocations' % resource)
        allocations = body['allocations']
        if sort_by:
            allocations = sorted(allocations, key=lambda l: l[sort_by])
        return allocations