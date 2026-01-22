from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_create_kwargs(self):
    keys = ('availability_zone', 'is_multiattach', 'size', 'name', 'description', 'volume_type', 'scheduler_hints', 'metadata')
    kwargs = {k: self.params[k] for k in keys if self.params[k] is not None}
    find_filters = {}
    if self.params['snapshot']:
        snapshot = self.conn.block_storage.find_snapshot(self.params['snapshot'], ignore_missing=False, **find_filters)
        kwargs['snapshot_id'] = snapshot.id
    if self.params['image']:
        image = self.conn.image.find_image(self.params['image'], ignore_missing=False)
        kwargs['image_id'] = image.id
    if self.params['volume']:
        volume = self.conn.block_storage.find_volume(self.params['volume'], ignore_missing=False, **find_filters)
        kwargs['source_volume_id'] = volume.id
    return kwargs