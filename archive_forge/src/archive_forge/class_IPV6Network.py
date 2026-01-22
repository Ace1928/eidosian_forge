import abc
from collections import abc as collections_abc
import datetime
from distutils import versionpredicate
import re
import uuid
import warnings
import copy
import iso8601
import netaddr
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import _utils
from oslo_versionedobjects import exception
class IPV6Network(IPNetwork):

    def __init__(self, *args, **kwargs):
        super(IPV6Network, self).__init__(*args, **kwargs)
        self.PATTERN = self._create_pattern()

    @staticmethod
    def coerce(obj, attr, value):
        try:
            return netaddr.IPNetwork(value, version=6)
        except netaddr.AddrFormatError as e:
            raise ValueError(str(e))

    def _create_pattern(self):
        ipv6seg = '[0-9a-fA-F]{1,4}'
        ipv4seg = '(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
        return '^(' + ipv6seg + ':){7,7}' + ipv6seg + '|(' + ipv6seg + ':){1,7}:|(' + ipv6seg + ':){1,6}:' + ipv6seg + '|(' + ipv6seg + ':){1,5}(:' + ipv6seg + '){1,2}|(' + ipv6seg + ':){1,4}(:' + ipv6seg + '){1,3}|(' + ipv6seg + ':){1,3}(:' + ipv6seg + '){1,4}|(' + ipv6seg + ':){1,2}(:' + ipv6seg + '){1,5}|' + ipv6seg + ':((:' + ipv6seg + '){1,6})|:((:' + ipv6seg + '){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}(' + ipv4seg + '\\.){3,3}' + ipv4seg + '|(' + ipv6seg + ':){1,4}:(' + ipv4seg + '\\.){3,3}' + ipv4seg + '(\\/(d|dd|1[0-1]d|12[0-8]))$'