from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMRecordSetInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(relative_name=dict(type='str'), resource_group=dict(type='str'), zone_name=dict(type='str'), record_type=dict(type='str'), top=dict(type='int'))
        self.results = dict(changed=False)
        self.relative_name = None
        self.resource_group = None
        self.zone_name = None
        self.record_type = None
        self.top = None
        super(AzureRMRecordSetInfo, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_dnsrecordset_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_dnsrecordset_facts' module has been renamed to 'azure_rm_dnsrecordset_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if not self.top or self.top <= 0:
            self.top = None
        if self.relative_name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name or record type.')
        if self.relative_name and (not self.zone_name):
            self.fail('Parameter error: DNS Zone required when filtering by name or record type.')
        results = []
        if self.relative_name is not None:
            results = self.get_item()
        elif self.record_type:
            results = self.list_type()
        elif self.zone_name:
            results = self.list_zone()
        if is_old_facts:
            self.results['ansible_facts'] = {'azure_dnsrecordset': self.serialize_list(results)}
        self.results['dnsrecordsets'] = self.curated_list(results)
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.relative_name))
        item = None
        results = []
        try:
            item = self.dns_client.record_sets.get(self.resource_group, self.zone_name, self.relative_name, self.record_type)
        except ResourceNotFoundError:
            results = []
            pass
        else:
            results = [item]
        return results

    def list_type(self):
        self.log('Lists the record sets of a specified type in a DNS zone')
        try:
            response = self.dns_client.record_sets.list_by_type(self.resource_group, self.zone_name, self.record_type, top=self.top)
        except Exception as exc:
            self.fail('Failed to list for record type {0} - {1}'.format(self.record_type, str(exc)))
        results = []
        for item in response:
            results.append(item)
        return results

    def list_zone(self):
        self.log('Lists all record sets in a DNS zone')
        try:
            response = self.dns_client.record_sets.list_by_dns_zone(self.resource_group, self.zone_name, top=self.top)
        except Exception as exc:
            self.fail('Failed to list for zone {0} - {1}'.format(self.zone_name, str(exc)))
        results = []
        for item in response:
            results.append(item)
        return results

    def serialize_list(self, raws):
        return [self.serialize_obj(item, AZURE_OBJECT_CLASS) for item in raws] if raws else []

    def curated_list(self, raws):
        return [self.record_to_dict(item) for item in raws] if raws else []

    def record_to_dict(self, record):
        record_type = record.type[len('Microsoft.Network/dnszones/'):]
        records = getattr(record, RECORDSET_VALUE_MAP.get(record_type))
        if records:
            if not isinstance(records, list):
                records = [records]
        else:
            records = []
        return dict(id=record.id, relative_name=record.name, record_type=record_type, records=[x.as_dict() for x in records], time_to_live=record.ttl, fqdn=record.fqdn, provisioning_state=record.provisioning_state, metadata=record.metadata)