from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMADApplicationInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(app_id=dict(type='str'), object_id=dict(type='str'), identifier_uri=dict(type='str'), tenant=dict(type='str', required=True))
        self.tenant = None
        self.app_id = None
        self.object_id = None
        self.identifier_uri = None
        self.results = dict(changed=False)
        super(AzureRMADApplicationInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, is_ad_resource=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        applications = []
        try:
            client = self.get_graphrbac_client(self.tenant)
            if self.object_id:
                applications = [client.applications.get(self.object_id)]
            else:
                sub_filters = []
                if self.identifier_uri:
                    sub_filters.append("identifierUris/any(s:s eq '{0}')".format(self.identifier_uri))
                if self.app_id:
                    sub_filters.append("appId eq '{0}'".format(self.app_id))
                applications = list(client.applications.list(filter=' and '.join(sub_filters)))
            self.results['applications'] = [self.to_dict(app) for app in applications]
        except GraphErrorException as ge:
            self.fail('failed to get application info {0}'.format(str(ge)))
        return self.results

    def to_dict(self, object):
        return dict(app_id=object.app_id, object_id=object.object_id, app_display_name=object.display_name, identifier_uris=object.identifier_uris)