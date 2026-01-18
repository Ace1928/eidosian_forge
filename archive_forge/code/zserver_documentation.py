import contextlib
import logging
import os
import socket
import struct
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.base.app_manager import OSKenApp
from os_ken.controller.handler import set_ev_cls
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.services.protocols.zebra import db
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.server import event as zserver_event

        Sends Zebra message.

        :param msg: Instance of py:class: `os_ken.lib.packet.zebra.ZebraMessage`.
        :return: Serialized msg if succeeded, otherwise None.
        