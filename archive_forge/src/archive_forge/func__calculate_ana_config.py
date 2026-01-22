from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.fc_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def _calculate_ana_config(self, want_ana, have_ana):
    """
        get the cmds based on want_ana and have_ana and the state

        Args:
            want_ana (str): analytics type which you want
            have_ana (str): analytics type which you have

        +----------+----------+---------+
        |            MERGED             |
        |----------+----------+---------+
        | want_ana | have_ana | outcome |
        +----------+----------+---------+
        | ""       | *        | no op   |
        | fc-scsi  | *        | fc-scsi |
        | fc-scsi  | fc-all   | no op   |
        | fc-nvme  | *        | fc-nvme |
        | fc-nvme  | fc-all   | no op   |
        +----------+----------+---------+


        +----------+----------+-----------+
        |            DELETED              |
        |----------+----------+-----------+
        | want_ana | have_ana | outcome   |
        +----------+----------+-----------+
        | *        | fc-scsi  | no fc-all |
        | *        | fc-nvme  | no fc-all |
        | *        | fc-all   | no fc-all |
        | *        | ""       | no op     |
        +----------+----------+-----------+


        +----------+----------+---------------------+
        |            REPLACED/OVERRIDEN             |
        |----------+----------+---------------------+
        | want_ana | have_ana | outcome             |
        +----------+----------+---------------------+
        | ""       | *        | no fc-all           |
        | fc-scsi  | ""       | fc-scsi             |
        | fc-nvme  | ""       | fc-nvme             |
        | fc-all   | ""       | fc-all              |
        | fc-scsi  | *        | no fc-all ; fc-scsi |
        | fc-nvme  | *        | no fc-all ; fc-nvme |
        | fc-all   | *        | fc-all              |
        +----------+----------+---------------------+


        """
    if want_ana == have_ana:
        return []
    val = []
    if self.state in ['overridden', 'replaced']:
        if want_ana == '':
            val = ['no analytics type fc-all']
        elif want_ana == 'fc-all':
            val = ['analytics type fc-all']
        elif have_ana == '':
            val = [f'analytics type {want_ana}']
        else:
            val = ['no analytics type fc-all', f'analytics type {want_ana}']
    elif self.state in ['deleted']:
        if have_ana:
            val = ['no analytics type fc-all']
    elif self.state in ['merged']:
        if want_ana:
            if have_ana != 'fc-all':
                val = [f'analytics type {want_ana}']
    return val