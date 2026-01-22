from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
class CinderVolumeAttachment(vb.BaseVolumeAttachment):
    """Resource for associating volume to instance.

    Resource for associating existing volume to instance. Also, the location
    where the volume is exposed on the instance can be specified.
    """
    PROPERTIES = INSTANCE_ID, VOLUME_ID, DEVICE = ('instance_uuid', 'volume_id', 'mountpoint')
    properties_schema = {INSTANCE_ID: properties.Schema(properties.Schema.STRING, _('The ID of the server to which the volume attaches.'), required=True, update_allowed=True), VOLUME_ID: properties.Schema(properties.Schema.STRING, _('The ID of the volume to be attached.'), required=True, update_allowed=True, constraints=[constraints.CustomConstraint('cinder.volume')]), DEVICE: properties.Schema(properties.Schema.STRING, _('The location where the volume is exposed on the instance. This assignment may not be honored and it is advised that the path /dev/disk/by-id/virtio-<VolumeId> be used instead.'), update_allowed=True)}

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        prg_attach = None
        prg_detach = None
        if prop_diff:
            volume_id = self.properties[self.VOLUME_ID]
            server_id = self.properties[self.INSTANCE_ID]
            prg_detach = progress.VolumeDetachProgress(server_id, volume_id, self.resource_id)
            prg_detach.called = self.client_plugin('nova').detach_volume(server_id, self.resource_id)
            if self.VOLUME_ID in prop_diff:
                volume_id = prop_diff.get(self.VOLUME_ID)
            device = self.properties[self.DEVICE] if self.properties[self.DEVICE] else None
            if self.DEVICE in prop_diff:
                device = prop_diff[self.DEVICE] if prop_diff[self.DEVICE] else None
            if self.INSTANCE_ID in prop_diff:
                server_id = prop_diff.get(self.INSTANCE_ID)
            prg_attach = progress.VolumeAttachProgress(server_id, volume_id, device)
        return (prg_detach, prg_attach)

    def check_update_complete(self, checkers):
        prg_detach, prg_attach = checkers
        if not (prg_detach and prg_attach):
            return True
        if not prg_detach.cinder_complete:
            prg_detach.cinder_complete = self.client_plugin().check_detach_volume_complete(prg_detach.vol_id, prg_detach.srv_id)
            return False
        if not prg_detach.nova_complete:
            prg_detach.nova_complete = self.client_plugin('nova').check_detach_volume_complete(prg_detach.srv_id, self.resource_id)
            return False
        if not prg_attach.called:
            prg_attach.called = self.client_plugin('nova').attach_volume(prg_attach.srv_id, prg_attach.vol_id, prg_attach.device)
            return False
        if not prg_attach.complete:
            prg_attach.complete = self.client_plugin().check_attach_volume_complete(prg_attach.vol_id)
            if prg_attach.complete:
                self.resource_id_set(prg_attach.called)
            return prg_attach.complete
        return True