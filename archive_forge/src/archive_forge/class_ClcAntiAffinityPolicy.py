from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class ClcAntiAffinityPolicy:
    clc = clc_sdk
    module = None

    def __init__(self, module):
        """
        Construct module
        """
        self.module = module
        self.policy_dict = {}
        if not CLC_FOUND:
            self.module.fail_json(msg=missing_required_lib('clc-sdk'), exception=CLC_IMP_ERR)
        if not REQUESTS_FOUND:
            self.module.fail_json(msg=missing_required_lib('requests'), exception=REQUESTS_IMP_ERR)
        if requests.__version__ and LooseVersion(requests.__version__) < LooseVersion('2.5.0'):
            self.module.fail_json(msg='requests library  version should be >= 2.5.0')
        self._set_user_agent(self.clc)

    @staticmethod
    def _define_module_argument_spec():
        """
        Define the argument spec for the ansible module
        :return: argument spec dictionary
        """
        argument_spec = dict(name=dict(required=True), location=dict(required=True), state=dict(default='present', choices=['present', 'absent']))
        return argument_spec

    def process_request(self):
        """
        Process the request - Main Code Path
        :return: Returns with either an exit_json or fail_json
        """
        p = self.module.params
        self._set_clc_credentials_from_env()
        self.policy_dict = self._get_policies_for_datacenter(p)
        if p['state'] == 'absent':
            changed, policy = self._ensure_policy_is_absent(p)
        else:
            changed, policy = self._ensure_policy_is_present(p)
        if hasattr(policy, 'data'):
            policy = policy.data
        elif hasattr(policy, '__dict__'):
            policy = policy.__dict__
        self.module.exit_json(changed=changed, policy=policy)

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

    def _get_policies_for_datacenter(self, p):
        """
        Get the Policies for a datacenter by calling the CLC API.
        :param p: datacenter to get policies from
        :return: policies in the datacenter
        """
        response = {}
        policies = self.clc.v2.AntiAffinity.GetAll(location=p['location'])
        for policy in policies:
            response[policy.name] = policy
        return response

    def _create_policy(self, p):
        """
        Create an Anti Affinity Policy using the CLC API.
        :param p: datacenter to create policy in
        :return: response dictionary from the CLC API.
        """
        try:
            return self.clc.v2.AntiAffinity.Create(name=p['name'], location=p['location'])
        except CLCException as ex:
            self.module.fail_json(msg='Failed to create anti affinity policy : {0}. {1}'.format(p['name'], ex.response_text))

    def _delete_policy(self, p):
        """
        Delete an Anti Affinity Policy using the CLC API.
        :param p: datacenter to delete a policy from
        :return: none
        """
        try:
            policy = self.policy_dict[p['name']]
            policy.Delete()
        except CLCException as ex:
            self.module.fail_json(msg='Failed to delete anti affinity policy : {0}. {1}'.format(p['name'], ex.response_text))

    def _policy_exists(self, policy_name):
        """
        Check to see if an Anti Affinity Policy exists
        :param policy_name: name of the policy
        :return: boolean of if the policy exists
        """
        if policy_name in self.policy_dict:
            return self.policy_dict.get(policy_name)
        return False

    def _ensure_policy_is_absent(self, p):
        """
        Makes sure that a policy is absent
        :param p: dictionary of policy name
        :return: tuple of if a deletion occurred and the name of the policy that was deleted
        """
        changed = False
        if self._policy_exists(policy_name=p['name']):
            changed = True
            if not self.module.check_mode:
                self._delete_policy(p)
        return (changed, None)

    def _ensure_policy_is_present(self, p):
        """
        Ensures that a policy is present
        :param p: dictionary of a policy name
        :return: tuple of if an addition occurred and the name of the policy that was added
        """
        changed = False
        policy = self._policy_exists(policy_name=p['name'])
        if not policy:
            changed = True
            policy = None
            if not self.module.check_mode:
                policy = self._create_policy(p)
        return (changed, policy)

    @staticmethod
    def _set_user_agent(clc):
        if hasattr(clc, 'SetRequestsSession'):
            agent_string = 'ClcAnsibleModule/' + __version__
            ses = requests.Session()
            ses.headers.update({'Api-Client': agent_string})
            ses.headers['User-Agent'] += ' ' + agent_string
            clc.SetRequestsSession(ses)