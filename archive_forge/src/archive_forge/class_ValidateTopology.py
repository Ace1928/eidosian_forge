from openstack import resource
class ValidateTopology(AutoAllocatedTopology):
    base_path = '/auto-allocated-topology/%(project)s?fields=dry-run'
    dry_run = resource.Body('dry_run')
    project = resource.URI('project')