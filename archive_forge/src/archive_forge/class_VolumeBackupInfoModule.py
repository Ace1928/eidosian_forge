from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class VolumeBackupInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(), volume=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['name'] if self.params[k] is not None))
        volume_name_or_id = self.params['volume']
        volume = None
        if volume_name_or_id:
            volume = self.conn.block_storage.find_volume(volume_name_or_id)
            if volume:
                kwargs['volume_id'] = volume.id
        if volume_name_or_id and (not volume):
            backups = []
        else:
            backups = [b.to_dict(computed=False) for b in self.conn.block_storage.backups(**kwargs)]
        self.exit_json(changed=False, volume_backups=backups)