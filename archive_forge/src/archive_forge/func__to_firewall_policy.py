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
def _to_firewall_policy(self, data):
    rules = []
    for item in data.get('rules', []):
        rule = CloudSigmaFirewallPolicyRule(action=item['action'], direction=item['direction'], ip_proto=item['ip_proto'], src_ip=item['src_ip'], src_port=item['src_port'], dst_ip=item['dst_ip'], dst_port=item['dst_port'], comment=item['comment'])
        rules.append(rule)
    policy = CloudSigmaFirewallPolicy(id=data['uuid'], name=data['name'], rules=rules)
    return policy