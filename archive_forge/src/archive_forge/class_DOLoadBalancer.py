from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOLoadBalancer(object):
    size_regions = {'ams2', 'nyc2', 'sfo1'}
    all_sizes = {'lb-small', 'lb-medium', 'lb-large'}
    default_size = 'lb-small'
    min_size_unit = 1
    max_size_unit = 100
    default_size_unit = 1

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.id = None
        self.name = self.module.params.get('name')
        self.region = self.module.params.get('region')
        if self.region in DOLoadBalancer.size_regions:
            self.module.params.pop('size_unit')
            size = self.module.params.get('size', None)
            if not size:
                self.module.fail_json(msg="Missing required 'size' parameter")
            elif size not in DOLoadBalancer.all_sizes:
                self.module.fail_json(msg="Invalid 'size' parameter '{0}', must be one of: {1}".format(size, ', '.join(DOLoadBalancer.all_sizes)))
        else:
            self.module.params.pop('size')
            size_unit = self.module.params.get('size_unit', None)
            if not size_unit:
                self.module.fail_json(msg="Missing required 'size_unit' parameter")
            elif size_unit < DOLoadBalancer.min_size_unit or size_unit > DOLoadBalancer.max_size_unit:
                self.module.fail_json(msg="Invalid 'size_unit' parameter '{0}', must be in range: {1}-{2}".format(size_unit, DOLoadBalancer.min_size_unit, DOLoadBalancer.max_size_unit))
        self.updates = []
        self.module.params.pop('oauth_token')
        self.wait = self.module.params.pop('wait', True)
        self.wait_timeout = self.module.params.pop('wait_timeout', 600)
        if self.module.params.get('project_name'):
            self.projects = DigitalOceanProjects(module, self.rest)

    def get_by_id(self):
        """Fetch an existing DigitalOcean Load Balancer (by id)
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/get_load_balancer
        """
        response = self.rest.get('load_balancers/{0}'.format(self.id))
        json_data = response.json
        if response.status_code == 200:
            lb = json_data.get('load_balancer', None)
            if lb is not None:
                self.lb = lb
                return lb
            else:
                self.module.fail_json(msg='Unexpected error; please file a bug: get_by_id')
        return None

    def get_by_name(self):
        """Fetch all existing DigitalOcean Load Balancers
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/list_all_load_balancers
        """
        page = 1
        while page is not None:
            response = self.rest.get('load_balancers?page={0}'.format(page))
            json_data = response.json
            if json_data is None:
                self.module.fail_json(msg='Empty response from the DigitalOcean API; please try again or open a bug if it never succeeds.')
            if response.status_code == 200:
                lbs = json_data.get('load_balancers', [])
                for lb in lbs:
                    name = lb.get('name', None)
                    if name == self.name:
                        region = lb.get('region', None)
                        if region is not None:
                            region_slug = region.get('slug', None)
                            if region_slug is not None:
                                if region_slug == self.region:
                                    self.lb = lb
                                    return lb
                                else:
                                    self.module.fail_json(msg='Cannot change load balancer region -- delete and re-create')
                            else:
                                self.module.fail_json(msg='Unexpected error; please file a bug: get_by_name')
                        else:
                            self.module.fail_json(msg='Unexpected error; please file a bug: get_by_name')
                if 'links' in json_data and 'pages' in json_data['links'] and ('next' in json_data['links']['pages']):
                    page += 1
                else:
                    page = None
            else:
                self.module.fail_json(msg='Unexpected error; please file a bug: get_by_name')
        return None

    def ensure_active(self):
        """Wait for the existing Load Balancer to be active"""
        end_time = time.monotonic() + self.wait_timeout
        while time.monotonic() < end_time:
            if self.get_by_id():
                status = self.lb.get('status', None)
                if status is not None:
                    if status == 'active':
                        return True
                else:
                    self.module.fail_json(msg='Unexpected error; please file a bug: ensure_active')
            else:
                self.module.fail_json(msg='Load Balancer {0} in {1} not found'.format(self.id, self.region))
            time.sleep(10)
        self.module.fail_json(msg='Timed out waiting for Load Balancer {0} in {1} to be active'.format(self.id, self.region))

    def is_same(self, found_lb):
        """Checks if exising Load Balancer is the same as requested"""
        check_attributes = ['droplet_ids', 'size', 'size_unit', 'forwarding_rules', 'health_check', 'sticky_sessions', 'redirect_http_to_https', 'enable_proxy_protocol', 'enable_backend_keepalive']
        lb_region = found_lb.get('region', None)
        if not lb_region:
            self.module.fail_json(msg='Unexpected error; please file a bug should this persist: empty load balancer region')
        lb_region_slug = lb_region.get('slug', None)
        if not lb_region_slug:
            self.module.fail_json(msg='Unexpected error; please file a bug should this persist: empty load balancer region slug')
        for attribute in check_attributes:
            if attribute == 'size' and lb_region_slug not in DOLoadBalancer.size_regions:
                continue
            if attribute == 'size_unit' and lb_region_slug in DOLoadBalancer.size_regions:
                continue
            if self.module.params.get(attribute, None) != found_lb.get(attribute, None):
                self.updates.append(attribute)
        vpc_uuid = self.lb.get('vpc_uuid', None)
        if vpc_uuid is not None:
            if vpc_uuid != found_lb.get('vpc_uuid', None):
                self.updates.append('vpc_uuid')
        if len(self.updates):
            return False
        else:
            return True

    def update(self):
        """Updates a DigitalOcean Load Balancer
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/update_load_balancer
        """
        request_params = dict(self.module.params)
        self.id = self.lb.get('id', None)
        self.name = self.lb.get('name', None)
        self.vpc_uuid = self.lb.get('vpc_uuid', None)
        if self.id is not None and self.name is not None and (self.vpc_uuid is not None):
            request_params['vpc_uuid'] = self.vpc_uuid
            response = self.rest.put('load_balancers/{0}'.format(self.id), data=request_params)
            json_data = response.json
            if response.status_code == 200:
                self.module.exit_json(changed=True, msg='Load Balancer {0} ({1}) in {2} updated: {3}'.format(self.name, self.id, self.region, ', '.join(self.updates)))
            else:
                self.module.fail_json(changed=False, msg='Error updating Load Balancer {0} ({1}) in {2}: {3}'.format(self.name, self.id, self.region, json_data['message']))
        else:
            self.module.fail_json(msg='Unexpected error; please file a bug: update')

    def create(self):
        """Creates a DigitalOcean Load Balancer
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/create_load_balancer
        """
        found_lb = self.get_by_name()
        if found_lb is not None:
            if not self.is_same(found_lb):
                if self.module.check_mode:
                    self.module.exit_json(changed=False, msg='Load Balancer {0} already exists in {1} (and needs changes)'.format(self.name, self.region), data={'load_balancer': found_lb})
                else:
                    self.update()
            else:
                self.module.exit_json(changed=False, msg='Load Balancer {0} already exists in {1} (and needs no changes)'.format(self.name, self.region), data={'load_balancer': found_lb})
        if self.module.check_mode:
            self.module.exit_json(changed=False, msg='Would create Load Balancer {0} in {1}'.format(self.name, self.region))
        request_params = dict(self.module.params)
        response = self.rest.post('load_balancers', data=request_params)
        json_data = response.json
        if response.status_code != 202:
            self.module.fail_json(msg='Failed creating Load Balancer {0} in {1}: {2}'.format(self.name, self.region, json_data['message']))
        lb = json_data.get('load_balancer', None)
        if lb is None:
            self.module.fail_json(msg='Unexpected error; please file a bug: create empty lb')
        self.id = lb.get('id', None)
        if self.id is None:
            self.module.fail_json(msg='Unexpected error; please file a bug: create missing id')
        if self.wait:
            self.ensure_active()
        project_name = self.module.params.get('project_name')
        if project_name:
            urn = 'do:loadbalancer:{0}'.format(self.id)
            assign_status, error_message, resources = self.projects.assign_to_project(project_name, urn)
            self.module.exit_json(changed=True, data=json_data, msg=error_message, assign_status=assign_status, resources=resources)
        else:
            self.module.exit_json(changed=True, data=json_data)

    def delete(self):
        """Deletes a DigitalOcean Load Balancer
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/delete_load_balancer
        """
        lb = self.get_by_name()
        if lb is not None:
            id = lb.get('id', None)
            name = lb.get('name', None)
            lb_region = lb.get('region', None)
            if not lb_region:
                self.module.fail_json(msg='Unexpected error; please file a bug: delete missing region')
            lb_region_slug = lb_region.get('slug', None)
            if id is None or name is None or lb_region_slug is None:
                self.module.fail_json(msg='Unexpected error; please file a bug: delete missing id, name, or region slug')
            else:
                response = self.rest.delete('load_balancers/{0}'.format(id))
                json_data = response.json
                if response.status_code == 204:
                    self.module.exit_json(changed=True, msg='Load Balancer {0} ({1}) in {2} deleted'.format(name, id, lb_region_slug))
                else:
                    message = json_data.get('message', 'Empty failure message from the DigitalOcean API!')
                    self.module.fail_json(changed=False, msg='Failed to delete Load Balancer {0} ({1}) in {2}: {3}'.format(name, id, lb_region_slug, message))
        else:
            self.module.fail_json(changed=False, msg='Load Balancer {0} not found in {1}'.format(self.name, self.region))