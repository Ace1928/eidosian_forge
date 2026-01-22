from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ImageModule(OpenStackModule):
    argument_spec = dict(checksum=dict(), container_format=dict(default='bare'), disk_format=dict(default='qcow2'), filename=dict(), id=dict(), is_protected=dict(type='bool', aliases=['protected']), is_public=dict(type='bool', default=False), kernel=dict(), min_disk=dict(type='int'), min_ram=dict(type='int'), name=dict(required=True), owner=dict(aliases=['project']), owner_domain=dict(aliases=['project_domain']), properties=dict(type='dict', default={}), ramdisk=dict(), state=dict(default='present', choices=['absent', 'present']), tags=dict(type='list', default=[], elements='str'), visibility=dict(choices=['public', 'private', 'shared', 'community']), volume=dict())
    module_kwargs = dict(mutually_exclusive=[('filename', 'volume'), ('visibility', 'is_public')])
    attr_params = ('id', 'name', 'filename', 'disk_format', 'container_format', 'wait', 'timeout', 'is_public', 'is_protected', 'min_disk', 'min_ram', 'volume', 'tags')

    def _resolve_visibility(self):
        """resolve a visibility value to be compatible with older versions"""
        if self.params['visibility']:
            return self.params['visibility']
        if self.params['is_public'] is not None:
            return 'public' if self.params['is_public'] else 'private'
        return None

    def _build_params(self, owner):
        params = {attr: self.params[attr] for attr in self.attr_params}
        if owner:
            params['owner_id'] = owner.id
        params['visibility'] = self._resolve_visibility()
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def _return_value(self, image_name_or_id):
        image = self.conn.image.find_image(image_name_or_id)
        if image:
            image = image.to_dict(computed=False)
        return image

    def _build_update(self, image):
        update_payload = {'visibility': self._resolve_visibility()}
        for k in ('is_protected', 'min_disk', 'min_ram'):
            update_payload[k] = self.params[k]
        for k in ('kernel', 'ramdisk'):
            if not self.params[k]:
                continue
            k_id = '{0}_id'.format(k)
            k_image = self.conn.image.find_image(name_or_id=self.params[k], ignore_missing=False)
            update_payload[k_id] = k_image.id
        update_payload = {k: v for k, v in update_payload.items() if v is not None and image[k] != v}
        for p, v in self.params['properties'].items():
            if p not in image or image[p] != v:
                update_payload[p] = v
        if self.params['tags'] and set(image['tags']) != set(self.params['tags']):
            update_payload['tags'] = self.params['tags']
        if self.params['id'] and self.params['name'] and (self.params['name'] != image['name']):
            update_payload['name'] = self.params['name']
        return update_payload

    def run(self):
        changed = False
        image_name_or_id = self.params['id'] or self.params['name']
        owner_name_or_id = self.params['owner']
        owner_domain_name_or_id = self.params['owner_domain']
        owner_filters = {}
        if owner_domain_name_or_id:
            owner_domain = self.conn.identity.find_domain(owner_domain_name_or_id)
            if owner_domain:
                owner_filters['domain_id'] = owner_domain.id
            else:
                owner_filters['domain_id'] = owner_domain_name_or_id
        owner = None
        if owner_name_or_id:
            owner = self.conn.identity.find_project(owner_name_or_id, ignore_missing=False, **owner_filters)
        image = None
        if image_name_or_id:
            image = self.conn.get_image(image_name_or_id, filters={k: self.params[k] for k in ['checksum'] if self.params[k] is not None})
        changed = False
        if self.params['state'] == 'present':
            attrs = self._build_params(owner)
            if not image:
                image = self.conn.create_image(**attrs)
                changed = True
                if not self.params['wait']:
                    self.exit_json(changed=changed, image=self._return_value(image.id))
            update_payload = self._build_update(image)
            if update_payload:
                self.conn.image.update_image(image.id, **update_payload)
                changed = True
            self.exit_json(changed=changed, image=self._return_value(image.id))
        elif self.params['state'] == 'absent' and image is not None:
            self.conn.delete_image(name_or_id=image['id'], wait=self.params['wait'], timeout=self.params['timeout'])
            changed = True
        self.exit_json(changed=changed)