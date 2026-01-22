from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import progress
from heat.engine import resource
from heat.engine import rsrc_defn
class BaseVolumeAttachment(resource.Resource):
    """Base Volume Attachment Manager."""
    default_client_name = 'cinder'

    def handle_create(self):
        server_id = self.properties[self.INSTANCE_ID]
        volume_id = self.properties[self.VOLUME_ID]
        dev = self.properties[self.DEVICE] if self.properties[self.DEVICE] else None
        attach_id = self.client_plugin('nova').attach_volume(server_id, volume_id, dev)
        self.resource_id_set(attach_id)
        return volume_id

    def check_create_complete(self, volume_id):
        return self.client_plugin().check_attach_volume_complete(volume_id)

    def handle_delete(self):
        prg = None
        if self.resource_id:
            server_id = self.properties[self.INSTANCE_ID]
            vol_id = self.properties[self.VOLUME_ID]
            prg = progress.VolumeDetachProgress(server_id, vol_id, self.resource_id)
            prg.called = self.client_plugin('nova').detach_volume(server_id, self.resource_id)
        return prg

    def check_delete_complete(self, prg):
        if prg is None:
            return True
        if not prg.called:
            prg.called = self.client_plugin('nova').detach_volume(prg.srv_id, self.resource_id)
            return False
        if not prg.cinder_complete:
            prg.cinder_complete = self.client_plugin().check_detach_volume_complete(prg.vol_id, prg.srv_id)
            return False
        if not prg.nova_complete:
            prg.nova_complete = self.client_plugin('nova').check_detach_volume_complete(prg.srv_id, prg.attach_id)
            return prg.nova_complete
        return True