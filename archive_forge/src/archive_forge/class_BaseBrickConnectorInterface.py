from oslo_utils import importutils
from os_brick.initiator import connector
class BaseBrickConnectorInterface(object):

    def __init__(self, *args, **kwargs):
        self.connection_info = kwargs.get('connection_info')
        self.root_helper = kwargs.get('root_helper')
        self.use_multipath = kwargs.get('use_multipath')
        self.conn = connector.InitiatorConnector.factory(self.connection_info['driver_volume_type'], self.root_helper, conn=self.connection_info, use_multipath=self.use_multipath)

    def connect_volume(self, volume):
        device = self.conn.connect_volume(self.connection_info)
        return device

    def disconnect_volume(self, device):
        self.conn.disconnect_volume(self.connection_info, device, force=True)

    def extend_volume(self):
        self.conn.extend_volume(self.connection_info)

    def yield_path(self, volume, volume_path):
        """
        This method returns the volume file path.

        The reason for it's implementation is to fix Bug#2000584. More
        information is added in the ScaleIO connector which makes actual
        use of it's implementation.
        """
        return volume_path