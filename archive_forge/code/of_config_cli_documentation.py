import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
set_logical_switch_config <peer> <logical switch> <key> <value>
        eg. set_logical_switch_config sw1 running LogicalSwitch7 lost-connection-behavior failStandaloneMode
        