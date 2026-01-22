from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
class DigitalOceanProjects:

    def __init__(self, module, rest):
        self.module = module
        self.rest = rest
        self.get_all_projects()

    def get_all_projects(self):
        """Fetches all projects."""
        self.projects = self.rest.get_paginated_data(base_url='projects?', data_key_name='projects')

    def get_default(self):
        """Fetches the default project.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        project -- project dictionary representation (or {} if error)
        """
        project = [project for project in self.projects if project.get('is_default', False)]
        if len(project) == 0:
            return ('Unexpected error; no default project found', {})
        if len(project) > 1:
            return ('Unexpected error; more than one default project', {})
        return ('', project[0])

    def get_by_id(self, id):
        """Fetches the project with the given id.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        project -- project dictionary representation (or {} if error)
        """
        project = [project for project in self.projects if project.get('id') == id]
        if len(project) == 0:
            return ('No project with id {0} found'.format(id), {})
        elif len(project) > 1:
            return ('Unexpected error; more than one project with the same id', {})
        return ('', project[0])

    def get_by_name(self, name):
        """Fetches the project with the given name.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        project -- project dictionary representation (or {} if error)
        """
        project = [project for project in self.projects if project.get('name') == name]
        if len(project) == 0:
            return ('No project with name {0} found'.format(name), {})
        elif len(project) > 1:
            return ('Unexpected error; more than one project with the same name', {})
        return ('', project[0])

    def get_resources_by_id(self, id):
        """Fetches the project resources with the given id.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        resources -- resources dictionary representation (or {} if error)
        """
        resources = self.rest.get_paginated_data(base_url='projects/{0}/resources?'.format(id), data_key_name='resources')
        return ('', dict(resources=resources))

    def get_resources_by_name(self, name):
        """Fetches the project resources with the given name.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        resources -- resources dictionary representation (or {} if error)
        """
        err_msg, project = self.get_by_name(name)
        if err_msg:
            return (err_msg, {})
        return self.get_resources_by_id(project.get('id'))

    def get_resources_of_default(self):
        """Fetches default project resources.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        resources -- resources dictionary representation (or {} if error)
        """
        err_msg, project = self.get_default()
        if err_msg:
            return (err_msg, {})
        return self.get_resources_by_id(project.get('id'))

    def assign_to_project(self, project_name, urn):
        """Assign resource (urn) to project (name).

        Keyword arguments:
        project_name -- project name to associate the resource with
        urn -- resource URN (has the form do:resource_type:resource_id)

        Returns:
        assign_status -- ok, not_found, assigned, already_assigned, service_down
        error_message -- assignment error message (empty on success)
        resources -- resources assigned (or {} if error)

        Notes:
        For URN examples, see https://docs.digitalocean.com/reference/api/api-reference/#tag/Project-Resources

        Projects resources are identified by uniform resource names or URNs.
        A valid URN has the following format: do:resource_type:resource_id.

        The following resource types are supported:
        Resource Type  | Example URN
        Database       | do:dbaas:83c7a55f-0d84-4760-9245-aba076ec2fb2
        Domain         | do:domain:example.com
        Droplet        | do:droplet:4126873
        Floating IP    | do:floatingip:192.168.99.100
        Kubernetes     | do:kubernetes:bd5f5959-5e1e-4205-a714-a914373942af
        Load Balancer  | do:loadbalancer:39052d89-8dd4-4d49-8d5a-3c3b6b365b5b
        Space          | do:space:my-website-assets
        Volume         | do:volume:6fc4c277-ea5c-448a-93cd-dd496cfef71f
        """
        error_message, project = self.get_by_name(project_name)
        if not project:
            return ('', error_message, {})
        project_id = project.get('id', None)
        if not project_id:
            return ('', 'Unexpected error; cannot find project id for {0}'.format(project_name), {})
        data = {'resources': [urn]}
        response = self.rest.post('projects/{0}/resources'.format(project_id), data=data)
        status_code = response.status_code
        json = response.json
        if status_code != 200:
            message = json.get('message', 'No error message returned')
            return ('', 'Unable to assign resource {0} to project {1} [HTTP {2}: {3}]'.format(urn, project_name, status_code, message), {})
        resources = json.get('resources', [])
        if len(resources) == 0:
            return ('', 'Unexpected error; no resources returned (but assignment was successful)', {})
        if len(resources) > 1:
            return ('', 'Unexpected error; more than one resource returned (but assignment was successful)', {})
        status = resources[0].get('status', 'Unexpected error; no status returned (but assignment was successful)')
        return (status, 'Assigned {0} to project {1}'.format(urn, project_name), resources[0])