import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
Manage a bridge
        :Parameters:
            - name: bridge name
            - subnet: network cider to be used in this bridge
            - start_ip: start address of an ip to be used in the subnet
            - end_ip: end address of an ip to be used in the subnet
            - with_ip: specify if assign automatically an ip address
            - self_ip: specify if assign an ip address for the bridge
            - fixed_ip: an ip address to be assigned to the bridge
            - reuse: specify if use an existing bridge
            - br_type: One either in a 'docker', 'brctl' or 'ovs'
        