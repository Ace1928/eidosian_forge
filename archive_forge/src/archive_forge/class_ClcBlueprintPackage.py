from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class ClcBlueprintPackage:
    clc = clc_sdk
    module = None

    def __init__(self, module):
        """
        Construct module
        """
        self.module = module
        if not CLC_FOUND:
            self.module.fail_json(msg=missing_required_lib('clc-sdk'), exception=CLC_IMP_ERR)
        if not REQUESTS_FOUND:
            self.module.fail_json(msg=missing_required_lib('requests'), exception=REQUESTS_IMP_ERR)
        if requests.__version__ and LooseVersion(requests.__version__) < LooseVersion('2.5.0'):
            self.module.fail_json(msg='requests library  version should be >= 2.5.0')
        self._set_user_agent(self.clc)

    def process_request(self):
        """
        Process the request - Main Code Path
        :return: Returns with either an exit_json or fail_json
        """
        p = self.module.params
        changed = False
        changed_server_ids = []
        self._set_clc_credentials_from_env()
        server_ids = p['server_ids']
        package_id = p['package_id']
        package_params = p['package_params']
        state = p['state']
        if state == 'present':
            changed, changed_server_ids, request_list = self.ensure_package_installed(server_ids, package_id, package_params)
            self._wait_for_requests_to_complete(request_list)
        self.module.exit_json(changed=changed, server_ids=changed_server_ids)

    @staticmethod
    def define_argument_spec():
        """
        This function defines the dictionary object required for
        package module
        :return: the package dictionary object
        """
        argument_spec = dict(server_ids=dict(type='list', elements='str', required=True), package_id=dict(required=True), package_params=dict(type='dict', default={}), wait=dict(default=True), state=dict(default='present', choices=['present']))
        return argument_spec

    def ensure_package_installed(self, server_ids, package_id, package_params):
        """
        Ensure the package is installed in the given list of servers
        :param server_ids: the server list where the package needs to be installed
        :param package_id: the blueprint package id
        :param package_params: the package arguments
        :return: (changed, server_ids, request_list)
                    changed: A flag indicating if a change was made
                    server_ids: The list of servers modified
                    request_list: The list of request objects from clc-sdk
        """
        changed = False
        request_list = []
        servers = self._get_servers_from_clc(server_ids, 'Failed to get servers from CLC')
        for server in servers:
            if not self.module.check_mode:
                request = self.clc_install_package(server, package_id, package_params)
                request_list.append(request)
            changed = True
        return (changed, server_ids, request_list)

    def clc_install_package(self, server, package_id, package_params):
        """
        Install the package to a given clc server
        :param server: The server object where the package needs to be installed
        :param package_id: The blue print package id
        :param package_params: the required argument dict for the package installation
        :return: The result object from the CLC API call
        """
        result = None
        try:
            result = server.ExecutePackage(package_id=package_id, parameters=package_params)
        except CLCException as ex:
            self.module.fail_json(msg='Failed to install package : {0} to server {1}. {2}'.format(package_id, server.id, ex.message))
        return result

    def _wait_for_requests_to_complete(self, request_lst):
        """
        Waits until the CLC requests are complete if the wait argument is True
        :param request_lst: The list of CLC request objects
        :return: none
        """
        if not self.module.params['wait']:
            return
        for request in request_lst:
            request.WaitUntilComplete()
            for request_details in request.requests:
                if request_details.Status() != 'succeeded':
                    self.module.fail_json(msg='Unable to process package install request')

    def _get_servers_from_clc(self, server_list, message):
        """
        Internal function to fetch list of CLC server objects from a list of server ids
        :param server_list: the list of server ids
        :param message: the error message to raise if there is any error
        :return the list of CLC server objects
        """
        try:
            return self.clc.v2.Servers(server_list).servers
        except CLCException as ex:
            self.module.fail_json(msg=message + ': %s' % ex)

    def _set_clc_credentials_from_env(self):
        """
        Set the CLC Credentials on the sdk by reading environment variables
        :return: none
        """
        env = os.environ
        v2_api_token = env.get('CLC_V2_API_TOKEN', False)
        v2_api_username = env.get('CLC_V2_API_USERNAME', False)
        v2_api_passwd = env.get('CLC_V2_API_PASSWD', False)
        clc_alias = env.get('CLC_ACCT_ALIAS', False)
        api_url = env.get('CLC_V2_API_URL', False)
        if api_url:
            self.clc.defaults.ENDPOINT_URL_V2 = api_url
        if v2_api_token and clc_alias:
            self.clc._LOGIN_TOKEN_V2 = v2_api_token
            self.clc._V2_ENABLED = True
            self.clc.ALIAS = clc_alias
        elif v2_api_username and v2_api_passwd:
            self.clc.v2.SetCredentials(api_username=v2_api_username, api_passwd=v2_api_passwd)
        else:
            return self.module.fail_json(msg='You must set the CLC_V2_API_USERNAME and CLC_V2_API_PASSWD environment variables')

    @staticmethod
    def _set_user_agent(clc):
        if hasattr(clc, 'SetRequestsSession'):
            agent_string = 'ClcAnsibleModule/' + __version__
            ses = requests.Session()
            ses.headers.update({'Api-Client': agent_string})
            ses.headers['User-Agent'] += ' ' + agent_string
            clc.SetRequestsSession(ses)