import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class AvId(enum.IntFlag):
    """ID for an NTLM AV_PAIR.

    These are the IDs that can be set as the `AvId` on an `AV_PAIR`_.

    .. _AV_PAIR:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/83f5e789-660d-4781-8491-5f8c6641f75e
    """
    eol = 0
    nb_computer_name = 1
    nb_domain_name = 2
    dns_computer_name = 3
    dns_domain_name = 4
    dns_tree_name = 5
    flags = 6
    timestamp = 7
    single_host = 8
    target_name = 9
    channel_bindings = 10

    @classmethod
    def native_labels(cls) -> typing.Dict['AvId', str]:
        return {AvId.eol: 'MSV_AV_EOL', AvId.nb_computer_name: 'MSV_AV_NB_COMPUTER_NAME', AvId.nb_domain_name: 'MSV_AV_NB_DOMAIN_NAME', AvId.dns_computer_name: 'MSV_AV_DNS_COMPUTER_NAME', AvId.dns_domain_name: 'MSV_AV_DNS_DOMAIN_NAME', AvId.dns_tree_name: 'MSV_AV_DNS_TREE_NAME', AvId.flags: 'MSV_AV_FLAGS', AvId.timestamp: 'MSV_AV_TIMESTAMP', AvId.single_host: 'MSV_AV_SINGLE_HOST', AvId.target_name: 'MSV_AV_TARGET_NAME', AvId.channel_bindings: 'MSV_AV_CHANNEL_BINDINGS'}