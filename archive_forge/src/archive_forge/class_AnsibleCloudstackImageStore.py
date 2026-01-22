from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
class AnsibleCloudstackImageStore(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudstackImageStore, self).__init__(module)
        self.returns = {'protocol': 'protocol', 'providername': 'provider_name', 'scope': 'scope', 'url': 'url'}
        self.image_store = None

    def get_storage_providers(self, storage_type='image'):
        args = {'type': storage_type}
        storage_provides = self.query_api('listStorageProviders', **args)
        return [provider.get('name') for provider in storage_provides.get('dataStoreProvider')]

    def get_image_store(self):
        if self.image_store:
            return self.image_store
        image_store = self.module.params.get('name')
        args = {'name': self.module.params.get('name'), 'zoneid': self.get_zone(key='id')}
        image_stores = self.query_api('listImageStores', **args)
        if image_stores:
            for img_s in image_stores.get('imagestore'):
                if image_store.lower() in [img_s['name'].lower(), img_s['id']]:
                    self.image_store = img_s
                    break
        return self.image_store

    def present_image_store(self):
        provider_list = self.get_storage_providers()
        image_store = self.get_image_store()
        if self.module.params.get('provider') not in provider_list:
            self.module.fail_json(msg='Provider %s is not in the provider list (%s). Please specify a correct provider' % (self.module.params.get('provider'), provider_list))
        args = {'name': self.module.params.get('name'), 'url': self.module.params.get('url'), 'zoneid': self.get_zone(key='id'), 'provider': self.module.params.get('provider')}
        if not image_store:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('addImageStore', **args)
                self.image_store = res.get('imagestore')
        else:
            args['providername'] = args.pop('provider')
            if self.has_changed(args, image_store):
                if self.module.params.get('force_recreate'):
                    self.absent_image_store()
                    self.image_store = None
                    self.image_store = self.present_image_store()
                else:
                    self.module.warn("Changes to the Image Store won't be appliedUse force_recreate=yes to allow the store to be recreated")
        return self.image_store

    def absent_image_store(self):
        image_store = self.get_image_store()
        if image_store:
            self.result['changed'] = True
            if not self.module.check_mode:
                args = {'id': image_store.get('id')}
                self.query_api('deleteImageStore', **args)
        return image_store