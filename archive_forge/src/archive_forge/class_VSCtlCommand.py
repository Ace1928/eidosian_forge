import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
class VSCtlCommand(StringifyMixin):
    """
    Class to describe artgumens similar to those of ``ovs-vsctl`` command.

    ``command`` specifies the command of ``ovs-vsctl``.

    ``args`` specifies a list or tuple of arguments for the given command.

    ``options`` specifies a list or tuple of options for the given command.
    Please note that NOT all options of ``ovs-vsctl`` are supported.
    For example, ``--id`` option is not yet supported.
    This class supports the followings.

    ================= =========================================================
    Option            Description
    ================= =========================================================
    ``--may-exist``   Does nothing when the given port already exists.
                      The supported commands are ``add-port`` and
                      ``add-bond``.
    ``--fake-iface``  Creates a port as a fake interface.
                      The supported command is ``add-bond``.
    ``--must-exist``  Raises exception if the given port does not exist.
                      The supported command is ``del-port``.
    ``--with-iface``  Takes effect to the interface which has the same name.
                      The supported command is ``del-port``.
    ``--if-exists``   Ignores exception when not found.
                      The supported command is ``get``.
    ================= =========================================================
    """

    def __init__(self, command, args=None, options=None):
        super(VSCtlCommand, self).__init__()
        self.command = command
        self.args = args or []
        self.options = options or []
        self.result = None
        self._prerequisite = None
        self._run = None

    def has_option(self, option):
        return option in self.options