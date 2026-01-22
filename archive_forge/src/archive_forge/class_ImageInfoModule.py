from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ImageInfoModule(OpenStackModule):
    argument_spec = dict(filters=dict(type='dict', aliases=['properties']), name=dict(aliases=['image']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['filters'] if self.params[k] is not None))
        name_or_id = self.params['name']
        if name_or_id is not None:
            kwargs['name_or_id'] = name_or_id
        self.exit(changed=False, images=[i.to_dict(computed=False) for i in self.conn.search_images(**kwargs)])