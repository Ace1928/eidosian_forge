import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class MultiHostDsn(AnyUrl):
    __slots__ = AnyUrl.__slots__ + ('hosts',)

    def __init__(self, *args: Any, hosts: Optional[List['HostParts']]=None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.hosts = hosts

    @staticmethod
    def _match_url(url: str) -> Optional[Match[str]]:
        return multi_host_url_regex().match(url)

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool=True) -> 'Parts':
        return super().validate_parts(parts, validate_port=False)

    @classmethod
    def _build_url(cls, m: Match[str], url: str, parts: 'Parts') -> 'MultiHostDsn':
        hosts_parts: List['HostParts'] = []
        host_re = host_regex()
        for host in m.groupdict()['hosts'].split(','):
            d: Parts = host_re.match(host).groupdict()
            host, tld, host_type, rebuild = cls.validate_host(d)
            port = d.get('port')
            cls._validate_port(port)
            hosts_parts.append({'host': host, 'host_type': host_type, 'tld': tld, 'rebuild': rebuild, 'port': port})
        if len(hosts_parts) > 1:
            return cls(None if any([hp['rebuild'] for hp in hosts_parts]) else url, scheme=parts['scheme'], user=parts['user'], password=parts['password'], path=parts['path'], query=parts['query'], fragment=parts['fragment'], host_type=None, hosts=hosts_parts)
        else:
            host_part = hosts_parts[0]
            return cls(None if host_part['rebuild'] else url, scheme=parts['scheme'], user=parts['user'], password=parts['password'], host=host_part['host'], tld=host_part['tld'], host_type=host_part['host_type'], port=host_part.get('port'), path=parts['path'], query=parts['query'], fragment=parts['fragment'])