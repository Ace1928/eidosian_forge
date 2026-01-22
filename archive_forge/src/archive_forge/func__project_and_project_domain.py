from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _project_and_project_domain(self):
    project_name_or_id = self.params['project']
    project_domain_name_or_id = self.params['project_domain']
    if project_domain_name_or_id:
        domain_id = self.conn.identity.find_domain(project_domain_name_or_id, ignore_missing=False).id
    else:
        domain_id = None
    kwargs = dict() if domain_id is None else dict(domain_id=domain_id)
    if project_name_or_id:
        project_id = self.conn.identity.find_project(project_name_or_id, *kwargs, ignore_missing=False).id
    else:
        project_id = None
    return (project_id, domain_id)