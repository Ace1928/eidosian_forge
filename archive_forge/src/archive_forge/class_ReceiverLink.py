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