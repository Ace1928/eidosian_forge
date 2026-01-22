from __future__ import absolute_import, division, print_function
import_status:
from ansible.module_utils.basic import (
from ansible.module_utils.urls import (
from ..module_utils.api import (
from ansible.module_utils._text import (
class AnsibleCloudscaleCustomImage(AnsibleCloudscaleBase):

    def _transform_import_to_image(self, imp):
        img = imp.get('custom_image', {})
        return {'href': img.get('href'), 'uuid': imp['uuid'], 'name': img.get('name'), 'created_at': None, 'size_gb': None, 'checksums': None, 'tags': imp['tags'], 'url': imp['url'], 'import_status': imp['status'], 'error_message': imp.get('error_message', ''), 'state': 'present', 'user_data_handling': self._module.params['user_data_handling'], 'zones': self._module.params['zones'], 'slug': self._module.params['slug'], 'firmware_type': self._module.params['firmware_type']}

    def _get_url(self, url):
        response, info = fetch_url(self._module, url, headers=self._auth_header, method='GET', timeout=self._module.params['api_timeout'])
        if info['status'] == 200:
            response = self._module.from_json(to_text(response.read(), errors='surrogate_or_strict'))
        elif info['status'] == 404:
            response = None
        elif info['status'] == 500 and url.startswith(self._api_url + self.resource_name + '/import/'):
            response = None
        else:
            self._module.fail_json(msg='Failure while calling the cloudscale.ch API with GET for "%s"' % url, fetch_url_info=info)
        return response

    def _get(self, api_call):
        api_url, call_uuid = api_call.split(self.resource_name)
        if not api_url:
            api_url = self._api_url
        response = self._get_url(api_url + self.resource_name + call_uuid) or []
        response_import = self._get_url(api_url + self.resource_name + '/import' + call_uuid) or []
        if call_uuid and response == [] and (response_import == []):
            return None
        if call_uuid and response:
            response = [response]
        if call_uuid and response_import:
            response_import = [response_import]
        response = dict([(i['uuid'], i) for i in response])
        response_import = dict([(i['uuid'], i) for i in response_import])
        response_import_filtered = dict([(k, v) for k, v in response_import.items() if v['status'] in ('success', 'in_progress')])
        import_names = set([v['custom_image']['name'] for k, v in response_import_filtered.items()])
        for k, v in reversed(list(response_import.items())):
            name = v['custom_image']['name']
            if v['status'] == 'failed' and name not in import_names:
                import_names.add(name)
                response_import_filtered[k] = v
        for uuid, imp in response_import_filtered.items():
            if uuid in response:
                response[uuid].update(url=imp['url'], import_status=imp['status'], error_message=imp.get('error_message', ''))
            else:
                response[uuid] = self._transform_import_to_image(imp)
        if not call_uuid:
            return response.values()
        else:
            return next(iter(response.values()))

    def _post(self, api_call, data=None):
        if not api_call.endswith('custom-images'):
            self._module.fail_json(msg='Error: Bad api_call URL.')
        api_call += '/import'
        if self._module.params['url']:
            return self._transform_import_to_image(self._post_or_patch('%s' % api_call, 'POST', data))
        else:
            self._module.fail_json(msg='Cannot import a new image without url.')

    def present(self):
        resource = self.query()
        if resource.get('firmware_type') is not None and resource.get('firmware_type') != self._module.params['firmware_type']:
            msg = 'Cannot change firmware type of an existing custom image'
            self._module.fail_json(msg)
        if resource['state'] == 'absent':
            resource = self.create(resource)
        elif resource.get('import_status') == 'failed' and (resource['url'] != self._module.params['url'] or self._module.params['force_retry']):
            resource = self.create(resource)
        else:
            resource = self.update(resource)
        return self.get_result(resource)