import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def add_static_route(self, network, next_hop):
    cmd = '/sbin/ip route add {0} via {1}'.format(network, next_hop)
    self.exec_on_ctn(cmd)