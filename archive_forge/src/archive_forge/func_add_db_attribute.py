import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def add_db_attribute(self, table, record, column, value, key=None):
    """
        Adds ('key'=)'value' into 'column' in 'record' in 'table'.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl add TBL REC COL [KEY=]VALUE
        """
    if key is not None:
        value = '%s=%s' % (key, value)
    command = ovs_vsctl.VSCtlCommand('add', (table, record, column, value))
    self.run_command([command])