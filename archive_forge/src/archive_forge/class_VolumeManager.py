from cinderclient.apiclient import base as common_base
from cinderclient import base
class VolumeManager(base.ManagerWithFind):
    """Manage :class:`Volume` resources."""
    resource_class = Volume

    def get(self, volume_id):
        """Get a volume.

        :param volume_id: The ID of the volume to get.
        :rtype: :class:`Volume`
        """
        return self._get('/volumes/%s' % volume_id, 'volume')

    def list(self, detailed=True, search_opts=None, marker=None, limit=None, sort=None):
        """Lists all volumes.

        :param detailed: Whether to return detailed volume info.
        :param search_opts: Search options to filter out volumes.
        :param marker: Begin returning volumes that appear later in the volume
                       list than that represented by this volume id.
        :param limit: Maximum number of volumes to return.
        :param sort: Sort information
        :rtype: list of :class:`Volume`
        """
        resource_type = 'volumes'
        url = self._build_list_url(resource_type, detailed=detailed, search_opts=search_opts, marker=marker, limit=limit, sort=sort)
        return self._list(url, resource_type, limit=limit)

    def delete(self, volume, cascade=False):
        """Delete a volume.

        :param volume: The :class:`Volume` to delete.
        :param cascade: Also delete dependent snapshots.
        """
        loc = '/volumes/%s' % base.getid(volume)
        if cascade:
            loc += '?cascade=True'
        return self._delete(loc)

    def update(self, volume, **kwargs):
        """Update the name or description for a volume.

        :param volume: The :class:`Volume` to update.
        """
        if not kwargs:
            return
        body = {'volume': kwargs}
        return self._update('/volumes/%s' % base.getid(volume), body)

    def _action(self, action, volume, info=None, **kwargs):
        """Perform a volume "action."

        :returns: tuple (response, body)
        """
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/volumes/%s/action' % base.getid(volume)
        resp, body = self.api.client.post(url, body=body)
        return common_base.TupleWithMeta((resp, body), resp)

    def attach(self, volume, instance_uuid, mountpoint, mode='rw', host_name=None):
        """Set attachment metadata.

        :param volume: The :class:`Volume` (or its ID)
                       you would like to attach.
        :param instance_uuid: uuid of the attaching instance.
        :param mountpoint: mountpoint on the attaching instance or host.
        :param mode: the access mode.
        :param host_name: name of the attaching host.
        """
        body = {'mountpoint': mountpoint, 'mode': mode}
        if instance_uuid is not None:
            body.update({'instance_uuid': instance_uuid})
        if host_name is not None:
            body.update({'host_name': host_name})
        return self._action('os-attach', volume, body)

    def detach(self, volume, attachment_uuid=None):
        """Clear attachment metadata.

        :param volume: The :class:`Volume` (or its ID)
                       you would like to detach.
        :param attachment_uuid: The uuid of the volume attachment.
        """
        return self._action('os-detach', volume, {'attachment_id': attachment_uuid})

    def reserve(self, volume):
        """Reserve this volume.

        :param volume: The :class:`Volume` (or its ID)
                       you would like to reserve.
        """
        return self._action('os-reserve', volume)

    def unreserve(self, volume):
        """Unreserve this volume.

        :param volume: The :class:`Volume` (or its ID)
                       you would like to unreserve.
        """
        return self._action('os-unreserve', volume)

    def begin_detaching(self, volume):
        """Begin detaching this volume.

        :param volume: The :class:`Volume` (or its ID)
                       you would like to detach.
        """
        return self._action('os-begin_detaching', volume)

    def roll_detaching(self, volume):
        """Roll detaching this volume.

        :param volume: The :class:`Volume` (or its ID)
                       you would like to roll detaching.
        """
        return self._action('os-roll_detaching', volume)

    def initialize_connection(self, volume, connector):
        """Initialize a volume connection.

        :param volume: The :class:`Volume` (or its ID).
        :param connector: connector dict from nova.
        """
        resp, body = self._action('os-initialize_connection', volume, {'connector': connector})
        return common_base.DictWithMeta(body['connection_info'], resp)

    def terminate_connection(self, volume, connector):
        """Terminate a volume connection.

        :param volume: The :class:`Volume` (or its ID).
        :param connector: connector dict from nova.
        """
        return self._action('os-terminate_connection', volume, {'connector': connector})

    def set_metadata(self, volume, metadata):
        """Update/Set a volumes metadata.

        :param volume: The :class:`Volume`.
        :param metadata: A list of keys to be set.
        """
        body = {'metadata': metadata}
        return self._create('/volumes/%s/metadata' % base.getid(volume), body, 'metadata')

    def delete_metadata(self, volume, keys):
        """Delete specified keys from volumes metadata.

        :param volume: The :class:`Volume`.
        :param keys: A list of keys to be removed.
        """
        response_list = []
        for k in keys:
            resp, body = self._delete('/volumes/%s/metadata/%s' % (base.getid(volume), k))
            response_list.append(resp)
        return common_base.ListWithMeta([], response_list)

    def set_image_metadata(self, volume, metadata):
        """Set a volume's image metadata.

        :param volume: The :class:`Volume`.
        :param metadata: keys and the values to be set with.
        :type metadata: dict
        """
        return self._action('os-set_image_metadata', volume, {'metadata': metadata})

    def delete_image_metadata(self, volume, keys):
        """Delete specified keys from volume's image metadata.

        :param volume: The :class:`Volume`.
        :param keys: A list of keys to be removed.
        """
        response_list = []
        for key in keys:
            resp, body = self._action('os-unset_image_metadata', volume, {'key': key})
            response_list.append(resp)
        return common_base.ListWithMeta([], response_list)

    def show_image_metadata(self, volume):
        """Show a volume's image metadata.

        :param volume : The :class: `Volume` where the image metadata
            associated.
        """
        return self._action('os-show_image_metadata', volume)

    def upload_to_image(self, volume, force, image_name, container_format, disk_format):
        """Upload volume to image service as image.

        :param volume: The :class:`Volume` to upload.
        """
        return self._action('os-volume_upload_image', volume, {'force': force, 'image_name': image_name, 'container_format': container_format, 'disk_format': disk_format})

    def force_delete(self, volume):
        """Delete the specified volume ignoring its current state.

        :param volume: The :class:`Volume` to force-delete.
        """
        return self._action('os-force_delete', base.getid(volume))

    def reset_state(self, volume, state, attach_status=None, migration_status=None):
        """Update the provided volume with the provided state.

        :param volume: The :class:`Volume` to set the state.
        :param state: The state of the volume to be set.
        :param attach_status: The attach_status of the volume to be set,
                              or None to keep the current status.
        :param migration_status: The migration_status of the volume to be set,
                                 or None to keep the current status.
        """
        body = {'status': state} if state else {}
        if attach_status:
            body.update({'attach_status': attach_status})
        if migration_status:
            body.update({'migration_status': migration_status})
        return self._action('os-reset_status', volume, body)

    def extend(self, volume, new_size):
        """Extend the size of the specified volume.

        :param volume: The UUID of the volume to extend.
        :param new_size: The requested size to extend volume to.
        """
        return self._action('os-extend', base.getid(volume), {'new_size': new_size})

    def get_encryption_metadata(self, volume_id):
        """
        Retrieve the encryption metadata from the desired volume.

        :param volume_id: the id of the volume to query
        :return: a dictionary of volume encryption metadata
        """
        metadata = self._get('/volumes/%s/encryption' % volume_id)
        return common_base.DictWithMeta(metadata._info, metadata.request_ids)

    def migrate_volume(self, volume, host, force_host_copy, lock_volume):
        """Migrate volume to new host.

        :param volume: The :class:`Volume` to migrate
        :param host: The destination host
        :param force_host_copy: Skip driver optimizations
        :param lock_volume: Lock the volume and guarantee the migration
                            to finish
        """
        return self._action('os-migrate_volume', volume, {'host': host, 'force_host_copy': force_host_copy, 'lock_volume': lock_volume})

    def migrate_volume_completion(self, old_volume, new_volume, error):
        """Complete the migration from the old volume to the temp new one.

        :param old_volume: The original :class:`Volume` in the migration
        :param new_volume: The new temporary :class:`Volume` in the migration
        :param error: Inform of an error to cause migration cleanup
        """
        new_volume_id = base.getid(new_volume)
        resp, body = self._action('os-migrate_volume_completion', old_volume, {'new_volume': new_volume_id, 'error': error})
        return common_base.DictWithMeta(body, resp)

    def update_all_metadata(self, volume, metadata):
        """Update all metadata of a volume.

        :param volume: The :class:`Volume`.
        :param metadata: A list of keys to be updated.
        """
        body = {'metadata': metadata}
        return self._update('/volumes/%s/metadata' % base.getid(volume), body)

    def update_readonly_flag(self, volume, flag):
        return self._action('os-update_readonly_flag', base.getid(volume), {'readonly': flag})

    def retype(self, volume, volume_type, policy):
        """Change a volume's type.

        :param volume: The :class:`Volume` to retype
        :param volume_type: New volume type
        :param policy: Policy for migration during the retype
        """
        return self._action('os-retype', volume, {'new_type': volume_type, 'migration_policy': policy})

    def set_bootable(self, volume, flag):
        return self._action('os-set_bootable', base.getid(volume), {'bootable': flag})

    def manage(self, host, ref, name=None, description=None, volume_type=None, availability_zone=None, metadata=None, bootable=False):
        """Manage an existing volume."""
        body = {'volume': {'host': host, 'ref': ref, 'name': name, 'description': description, 'volume_type': volume_type, 'availability_zone': availability_zone, 'metadata': metadata, 'bootable': bootable}}
        return self._create('/os-volume-manage', body, 'volume')

    def unmanage(self, volume):
        """Unmanage a volume."""
        return self._action('os-unmanage', volume, None)

    def get_pools(self, detail):
        """Show pool information for backends."""
        query_string = ''
        if detail:
            query_string = '?detail=True'
        return self._get('/scheduler-stats/get_pools%s' % query_string, None)