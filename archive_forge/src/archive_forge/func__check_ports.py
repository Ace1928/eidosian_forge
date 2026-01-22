import os
import signal
import time
import fixtures
from pifpaf.drivers import rabbitmq
from oslo_messaging.tests.functional import utils
from oslo_messaging.tests import utils as test_utils
def _check_ports(self, port):
    rpc_server = self.servers.servers[0].server
    connection_contexts = [rpc_server.listener._poll_style_listener.conn, self.client.client.transport._driver._get_connection(), self.client.client.transport._driver._reply_q_conn]
    ports = [cctxt.connection.channel.connection.sock.getpeername()[1] for cctxt in connection_contexts]
    self.assertEqual([port] * len(ports), ports, 'expected: %s, rpc-server: %s, rpc-client: %s, rpc-replies: %s' % tuple([port] + ports))