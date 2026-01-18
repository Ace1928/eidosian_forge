import json
import ddt
from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
Set and unset node target RAID config data.

        Test steps:
        1) Create baremetal node in setUp.
        2) Set target RAID config data for the node
        3) Check target_raid_config of node equals to expected value.
        4) Unset target_raid_config data.
        5) Check that target_raid_config data is empty.
        