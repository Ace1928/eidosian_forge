from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.network.base import Network, NetworkCollector

    HP-UX-specifig subclass of Network. Defines networking facts:
    - default_interface
    - interfaces (a list of interface names)
    - interface_<name> dictionary of ipv4 address information.
    