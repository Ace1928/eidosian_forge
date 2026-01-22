from __future__ import annotations
from typing import Any, Optional  # noqa: H301
from oslo_log import log as logging
from oslo_utils import netutils
def _sanitize_host(host: str) -> str:
    if netutils.is_valid_ipv6(host):
        host = '[%s]' % host
    return host