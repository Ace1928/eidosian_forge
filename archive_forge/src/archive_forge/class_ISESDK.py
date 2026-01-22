from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
class ISESDK(object):

    def __init__(self, params):
        self.result = dict(changed=False, result='')
        if ISE_SDK_IS_INSTALLED:
            ise_uses_api_gateway = params.get('ise_uses_api_gateway')
            ui_base_url = None
            ers_base_url = None
            mnt_base_url = None
            px_grid_base_url = None
            if not ise_uses_api_gateway:
                ui_base_url = get_ise_url(params.get('ise_hostname'), port='443')
                ers_base_url = get_ise_url(params.get('ise_hostname'), port='9060')
                mnt_base_url = get_ise_url(params.get('ise_hostname'), port='443')
                px_grid_base_url = get_ise_url(params.get('ise_hostname'), port='8910')
            self.api = api.IdentityServicesEngineAPI(username=params.get('ise_username'), password=params.get('ise_password'), base_url=get_ise_url(params.get('ise_hostname'), port=None), ui_base_url=ui_base_url, ers_base_url=ers_base_url, mnt_base_url=mnt_base_url, px_grid_base_url=px_grid_base_url, single_request_timeout=params.get('ise_single_request_timeout'), verify=params.get('ise_verify'), version=params.get('ise_version'), wait_on_rate_limit=params.get('ise_wait_on_rate_limit'), uses_api_gateway=ise_uses_api_gateway, uses_csrf_token=params.get('ise_uses_csrf_token'), debug=params.get('ise_debug'))
            if params.get('ise_debug') and LOGGING_IN_STANDARD:
                logging.getLogger('ciscoisesdk').addHandler(logging.StreamHandler())
        else:
            self.fail_json(msg="Cisco ISE Python SDK is not installed. Execute 'pip install ciscoisesdk'")

    def changed(self):
        self.result['changed'] = True

    def object_created(self):
        self.changed()
        self.result['result'] = 'Object created'

    def object_updated(self):
        self.changed()
        self.result['result'] = 'Object updated'

    def object_deleted(self):
        self.changed()
        self.result['result'] = 'Object deleted'

    def object_already_absent(self):
        self.result['result'] = 'Object already absent'

    def object_already_present(self):
        self.result['result'] = 'Object already present'

    def object_present_and_different(self):
        self.result['result'] = 'Object already present, but it has different values to the requested'

    def object_modify_result(self, changed=None, result=None):
        if result is not None:
            self.result['result'] = result
        if changed:
            self.changed()

    def exec(self, family, function, params=None, handle_func_exception=True):
        try:
            family = getattr(self.api, family)
            func = getattr(family, function)
        except Exception as e:
            self.fail_json(msg='An error occured when retrieving operation. The error was: {error}'.format(error=e))
        try:
            if params:
                response = func(**params)
            else:
                response = func()
        except exceptions.ciscoisesdkException as e:
            if handle_func_exception:
                self.fail_json(msg='An error occured when executing operation. The error was: {error}'.format(error=e))
            else:
                raise e
        return response

    def fail_json(self, msg, **kwargs):
        self.result.update(**kwargs)
        raise AnsibleActionFail(msg, kwargs)

    def exit_json(self):
        return self.result