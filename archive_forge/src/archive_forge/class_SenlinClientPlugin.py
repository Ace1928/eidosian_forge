from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
class SenlinClientPlugin(sdk_plugin.OpenStackSDKPlugin):
    exceptions_module = exceptions

    def _create(self, version=None):
        client = super(SenlinClientPlugin, self)._create(version=version)
        return client.clustering

    def _get_additional_create_args(self, version):
        return {'clustering_api_version': version or '1'}

    def generate_spec(self, spec_type, spec_props):
        spec = {'properties': spec_props}
        spec['type'], spec['version'] = spec_type.split('-')
        return spec

    def check_action_status(self, action_id):
        action = self.client().get_action(action_id)
        if action.status == 'SUCCEEDED':
            return True
        elif action.status == 'FAILED':
            raise exception.ResourceInError(status_reason=action.status_reason, resource_status=action.status)
        return False

    def cluster_is_active(self, cluster_id):
        cluster = self.client().get_cluster(cluster_id)
        if cluster.status == 'ACTIVE':
            return True
        elif cluster.status == 'ERROR':
            raise exception.ResourceInError(status_reason=cluster.status_reason, resource_status=cluster.status)
        return False

    def get_profile_id(self, profile_name):
        profile = self.client().get_profile(profile_name)
        return profile.id

    def get_cluster_id(self, cluster_name):
        cluster = self.client().get_cluster(cluster_name)
        return cluster.id

    def get_policy_id(self, policy_name):
        policy = self.client().get_policy(policy_name)
        return policy.id

    def is_bad_request(self, ex):
        return isinstance(ex, exceptions.HttpException) and ex.status_code == 400

    def execute_actions(self, actions):
        all_executed = True
        for action in actions:
            if action['done']:
                continue
            all_executed = False
            if 'action_id' in action:
                if action['action_id'] is None:
                    func = getattr(self.client(), action['func'])
                    ret = func(**action['params'])
                    if isinstance(ret, dict):
                        action['action_id'] = ret['action']
                    else:
                        action['action_id'] = ret.location.split('/')[-1]
                else:
                    ret = self.check_action_status(action['action_id'])
                    action['done'] = ret
            else:
                ret = self.cluster_is_active(action['cluster_id'])
                action['done'] = ret
            break
        return all_executed