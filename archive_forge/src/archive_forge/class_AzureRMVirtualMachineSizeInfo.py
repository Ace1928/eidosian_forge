from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMVirtualMachineSizeInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(location=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False, sizes=[])
        self.location = None
        self.name = None
        super(AzureRMVirtualMachineSizeInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.results['sizes'] = self.list_items_by_location()
        return self.results

    def list_items_by_location(self):
        self.log('List items by location')
        try:
            items = self.compute_client.virtual_machine_sizes.list(location=self.location)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list items - {0}'.format(str(exc)))
        return [self.serialize_size(item) for item in items if self.name is None or self.name == item.name]

    def serialize_size(self, size):
        """
        Convert a VirtualMachineSize object to dict.

        :param size: VirtualMachineSize object
        :return: dict
        """
        return self.serialize_obj(size, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES)