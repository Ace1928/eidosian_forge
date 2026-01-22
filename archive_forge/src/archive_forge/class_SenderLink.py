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