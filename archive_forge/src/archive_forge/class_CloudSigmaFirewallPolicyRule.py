import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigmaFirewallPolicyRule:
    """
    Represents a CloudSigma firewall policy rule.
    """

    def __init__(self, action, direction, ip_proto=None, src_ip=None, src_port=None, dst_ip=None, dst_port=None, comment=None):
        """
        :param action: Action (drop / accept).
        :type action: ``str``

        :param direction: Rule direction (in / out / both)>
        :type direction: ``str``

        :param ip_proto: IP protocol (tcp / udp).
        :type ip_proto: ``str``.

        :param src_ip: Source IP in CIDR notation.
        :type src_ip: ``str``

        :param src_port: Source port or a port range.
        :type src_port: ``str``

        :param dst_ip: Destination IP in CIDR notation.
        :type dst_ip: ``str``

        :param src_port: Destination port or a port range.
        :type src_port: ``str``

        :param comment: Comment associated with the policy.
        :type comment: ``str``
        """
        self.action = action
        self.direction = direction
        self.ip_proto = ip_proto
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.comment = comment

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<CloudSigmaFirewallPolicyRule action=%s, direction=%s>' % (self.action, self.direction)