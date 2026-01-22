import logging
from os_ken.services.protocols.bgp.net_ctrl import NET_CONTROLLER
from os_ken.services.protocols.bgp.net_ctrl import NOTIFICATION_LOG
class RpcLogHandler(logging.Handler):
    """Outputs log records to `NET_CONTROLLER`."""

    def emit(self, record):
        msg = self.format(record)
        NET_CONTROLLER.send_rpc_notification(NOTIFICATION_LOG, {'level': record.levelname, 'msg': msg})