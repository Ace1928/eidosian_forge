import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class FakeBroker(threading.Thread):
    """A test AMQP message 'broker'."""
    if pyngus:

        class Connection(pyngus.ConnectionEventHandler):
            """A single AMQP connection."""

            def __init__(self, server, socket_, name, product, sasl_mechanisms, user_credentials, sasl_config_dir, sasl_config_name):
                """Create a Connection using socket_."""
                self.socket = socket_
                self.name = name
                self.server = server
                self.sasl_mechanisms = sasl_mechanisms
                self.user_credentials = user_credentials
                properties = {'x-server': True}
                if self.sasl_mechanisms:
                    properties['x-sasl-mechs'] = self.sasl_mechanisms
                    if 'ANONYMOUS' not in self.sasl_mechanisms:
                        properties['x-require-auth'] = True
                if sasl_config_dir:
                    properties['x-sasl-config-dir'] = sasl_config_dir
                if sasl_config_name:
                    properties['x-sasl-config-name'] = sasl_config_name
                if self.server._ssl_config:
                    ssl = self.server._ssl_config
                    properties['x-ssl-server'] = True
                    properties['x-ssl-identity'] = (ssl['s_cert'], ssl['s_key'], ssl['pw'])
                    if ssl.get('authenticate_client'):
                        properties['x-ssl-ca-file'] = ssl['ca_cert']
                        properties['x-ssl-verify-mode'] = 'verify-peer'
                        properties['x-ssl-peer-name'] = ssl['c_name']
                if product:
                    properties['properties'] = {'product': product}
                self.connection = server.container.create_connection(name, self, properties)
                self.connection.user_context = self
                if pyngus.VERSION < (2, 0, 0):
                    if sasl_mechanisms:
                        self.connection.pn_sasl.mechanisms(sasl_mechanisms)
                        self.connection.pn_sasl.server()
                self.connection.open()
                self.sender_links = set()
                self.receiver_links = set()
                self.dead_links = set()

            def destroy(self):
                """Destroy the test connection."""
                for link in self.sender_links | self.receiver_links:
                    link.destroy()
                self.sender_links.clear()
                self.receiver_links.clear()
                self.dead_links.clear()
                self.connection.destroy()
                self.connection = None
                self.socket.close()
                self.socket = None

            def fileno(self):
                """Allows use of this in a select() call."""
                return self.socket.fileno()

            def process_input(self):
                """Called when socket is read-ready."""
                try:
                    pyngus.read_socket_input(self.connection, self.socket)
                    self.connection.process(time.time())
                except socket.error:
                    self._socket_error()

            def send_output(self):
                """Called when socket is write-ready."""
                try:
                    pyngus.write_socket_output(self.connection, self.socket)
                    self.connection.process(time.time())
                except socket.error:
                    self._socket_error()

            def _socket_error(self):
                self.connection.close_input()
                self.connection.close_output()

            def connection_active(self, connection):
                self.server.connection_count += 1

            def connection_remote_closed(self, connection, reason):
                """Peer has closed the connection."""
                self.connection.close()

            def connection_closed(self, connection):
                """Connection close completed."""
                self.server.connection_count -= 1

            def connection_failed(self, connection, error):
                """Connection failure detected."""
                self.connection_closed(connection)

            def sender_requested(self, connection, link_handle, name, requested_source, properties):
                """Create a new message source."""
                addr = requested_source or 'source-' + uuid.uuid4().hex
                link = FakeBroker.SenderLink(self.server, self, link_handle, addr)
                self.sender_links.add(link)

            def receiver_requested(self, connection, link_handle, name, requested_target, properties):
                """Create a new message consumer."""
                addr = requested_target or 'target-' + uuid.uuid4().hex
                FakeBroker.ReceiverLink(self.server, self, link_handle, addr)

            def sasl_step(self, connection, pn_sasl):
                if 'PLAIN' in self.sasl_mechanisms:
                    credentials = pn_sasl.recv()
                    if not credentials:
                        return
                    if credentials not in self.user_credentials:
                        return pn_sasl.done(pn_sasl.AUTH)
                pn_sasl.done(pn_sasl.OK)

        class SenderLink(pyngus.SenderEventHandler):
            """An AMQP sending link."""

            def __init__(self, server, conn, handle, src_addr=None):
                self.server = server
                self.conn = conn
                cnn = conn.connection
                self.link = cnn.accept_sender(handle, source_override=src_addr, event_handler=self)
                conn.sender_links.add(self)
                self.link.open()
                self.routed = False

            def destroy(self):
                """Destroy the link."""
                conn = self.conn
                self.conn = None
                conn.sender_links.remove(self)
                conn.dead_links.discard(self)
                if self.link:
                    self.link.destroy()
                    self.link = None

            def send_message(self, message):
                """Send a message over this link."""

                def pyngus_callback(link, handle, state, info):
                    if state == pyngus.SenderLink.ACCEPTED:
                        self.server.sender_link_ack_count += 1
                    elif state == pyngus.SenderLink.RELEASED:
                        self.server.sender_link_requeue_count += 1
                self.link.send(message, delivery_callback=pyngus_callback)

            def _cleanup(self):
                if self.routed:
                    self.server.remove_route(self.link.source_address, self)
                    self.routed = False
                self.conn.dead_links.add(self)

            def sender_active(self, sender_link):
                self.server.sender_link_count += 1
                self.server.add_route(self.link.source_address, self)
                self.routed = True
                self.server.on_sender_active(sender_link)

            def sender_remote_closed(self, sender_link, error):
                self.link.close()

            def sender_closed(self, sender_link):
                self.server.sender_link_count -= 1
                self._cleanup()

            def sender_failed(self, sender_link, error):
                self.sender_closed(sender_link)

        class ReceiverLink(pyngus.ReceiverEventHandler):
            """An AMQP Receiving link."""

            def __init__(self, server, conn, handle, addr=None):
                self.server = server
                self.conn = conn
                cnn = conn.connection
                self.link = cnn.accept_receiver(handle, target_override=addr, event_handler=self)
                conn.receiver_links.add(self)
                self.link.open()

            def destroy(self):
                """Destroy the link."""
                conn = self.conn
                self.conn = None
                conn.receiver_links.remove(self)
                conn.dead_links.discard(self)
                if self.link:
                    self.link.destroy()
                    self.link = None

            def receiver_active(self, receiver_link):
                self.server.receiver_link_count += 1
                self.server.on_receiver_active(receiver_link)

            def receiver_remote_closed(self, receiver_link, error):
                self.link.close()

            def receiver_closed(self, receiver_link):
                self.server.receiver_link_count -= 1
                self.conn.dead_links.add(self)

            def receiver_failed(self, receiver_link, error):
                self.receiver_closed(receiver_link)

            def message_received(self, receiver_link, message, handle):
                """Forward this message out the proper sending link."""
                self.server.on_message(message, handle, receiver_link)
                if self.link.capacity < 1:
                    self.server.on_credit_exhausted(self.link)

    def __init__(self, cfg, sock_addr='', sock_port=0, product=None, default_exchange='Test-Exchange', sasl_mechanisms='ANONYMOUS', user_credentials=None, sasl_config_dir=None, sasl_config_name=None, ssl_config=None):
        """Create a fake broker listening on sock_addr:sock_port."""
        if not pyngus:
            raise AssertionError('pyngus module not present')
        threading.Thread.__init__(self)
        self._product = product
        self._sasl_mechanisms = sasl_mechanisms
        self._sasl_config_dir = sasl_config_dir
        self._sasl_config_name = sasl_config_name
        self._user_credentials = user_credentials
        self._ssl_config = ssl_config
        self._wakeup_pipe = os.pipe()
        self._my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._my_socket.bind((sock_addr, sock_port))
        self.host, self.port = self._my_socket.getsockname()
        self.container = pyngus.Container('test_server_%s:%d' % (self.host, self.port))
        af = AddresserFactory(default_exchange, cfg.addressing_mode, legacy_server_prefix=cfg.server_request_prefix, legacy_broadcast_prefix=cfg.broadcast_prefix, legacy_group_prefix=cfg.group_request_prefix, rpc_prefix=cfg.rpc_address_prefix, notify_prefix=cfg.notify_address_prefix, multicast=cfg.multicast_address, unicast=cfg.unicast_address, anycast=cfg.anycast_address)
        props = {'product': product} if product else {}
        self._addresser = af(props)
        self._connections = {}
        self._sources = {}
        self._pause = eventletutils.Event()
        self.direct_count = 0
        self.topic_count = 0
        self.fanout_count = 0
        self.fanout_sent_count = 0
        self.dropped_count = 0
        self.connection_count = 0
        self.sender_link_count = 0
        self.receiver_link_count = 0
        self.sender_link_ack_count = 0
        self.sender_link_requeue_count = 0
        self.message_log = []
        self.on_sender_active = lambda link: None
        self.on_receiver_active = lambda link: link.add_capacity(10)
        self.on_credit_exhausted = lambda link: link.add_capacity(10)
        self.on_message = lambda message, handle, link: self.forward_message(message, handle, link)

    def start(self):
        """Start the server."""
        LOG.debug('Starting Test Broker on %s:%d', self.host, self.port)
        self._shutdown = False
        self._closing = False
        self.daemon = True
        self._pause.set()
        self._my_socket.listen(10)
        super(FakeBroker, self).start()

    def pause(self):
        self._pause.clear()
        os.write(self._wakeup_pipe[1], b'!')

    def unpause(self):
        self._pause.set()

    def stop(self, clean=False):
        """Stop the server."""
        LOG.debug('Stopping test Broker %s:%d', self.host, self.port)
        if clean:
            self._closing = 1
        else:
            self._shutdown = True
        self._pause.set()
        os.write(self._wakeup_pipe[1], b'!')
        self.join()
        LOG.debug('Test Broker %s:%d stopped', self.host, self.port)

    def run(self):
        """Process I/O and timer events until the broker is stopped."""
        LOG.debug('Test Broker on %s:%d started', self.host, self.port)
        while not self._shutdown:
            self._pause.wait()
            readers, writers, timers = self.container.need_processing()
            readfd = [c.user_context for c in readers]
            readfd.extend([self._my_socket, self._wakeup_pipe[0]])
            writefd = [c.user_context for c in writers]
            timeout = None
            if timers:
                deadline = timers[0].next_tick
                now = time.time()
                timeout = 0 if deadline <= now else deadline - now
            readable, writable, ignore = select.select(readfd, writefd, [], timeout)
            worked = set()
            for r in readable:
                if r is self._my_socket:
                    sock, addr = self._my_socket.accept()
                    if not self._closing:
                        name = str(addr)
                        conn = FakeBroker.Connection(self, sock, name, self._product, self._sasl_mechanisms, self._user_credentials, self._sasl_config_dir, self._sasl_config_name)
                        self._connections[conn.name] = conn
                    else:
                        sock.close()
                elif r is self._wakeup_pipe[0]:
                    os.read(self._wakeup_pipe[0], 512)
                else:
                    r.process_input()
                    worked.add(r)
            for t in timers:
                now = time.time()
                if t.next_tick > now:
                    break
                t.process(now)
                conn = t.user_context
                worked.add(conn)
            for w in writable:
                w.send_output()
                worked.add(w)
            while worked:
                conn = worked.pop()
                if conn.connection.closed:
                    del self._connections[conn.name]
                    conn.destroy()
                else:
                    while conn.dead_links:
                        conn.dead_links.pop().destroy()
            if self._closing and (not self._connections):
                self._shutdown = True
            elif self._closing == 1:
                self._closing = 2
                for conn in self._connections.values():
                    conn.connection.close()
        self._my_socket.close()
        for conn in self._connections.values():
            conn.destroy()
        self._connections = None
        self.container.destroy()
        self.container = None
        return 0

    def add_route(self, address, link):
        if address not in self._sources:
            self._sources[address] = [link]
        elif link not in self._sources[address]:
            self._sources[address].append(link)

    def remove_route(self, address, link):
        if address in self._sources:
            if link in self._sources[address]:
                self._sources[address].remove(link)
                if not self._sources[address]:
                    del self._sources[address]

    def forward_message(self, message, handle, rlink):
        self.message_log.append(message)
        dest = message.address
        if dest not in self._sources:
            self.dropped_count += 1
            if '!no-ack!' not in dest:
                rlink.message_released(handle)
            return
        LOG.debug('Forwarding [%s]', dest)
        if self._addresser._is_multicast(dest):
            self.fanout_count += 1
            for link in self._sources[dest]:
                self.fanout_sent_count += 1
                LOG.debug('Broadcast to %s', dest)
                link.send_message(message)
        elif self._addresser._is_anycast(dest):
            self.topic_count += 1
            link = self._sources[dest].pop(0)
            link.send_message(message)
            LOG.debug('Send to %s', dest)
            self._sources[dest].append(link)
        else:
            self.direct_count += 1
            LOG.debug('Unicast to %s', dest)
            self._sources[dest][0].send_message(message)
        rlink.message_accepted(handle)