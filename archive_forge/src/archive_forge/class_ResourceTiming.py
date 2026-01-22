from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class ResourceTiming:
    """
    Timing information for the request.
    """
    request_time: float
    proxy_start: float
    proxy_end: float
    dns_start: float
    dns_end: float
    connect_start: float
    connect_end: float
    ssl_start: float
    ssl_end: float
    worker_start: float
    worker_ready: float
    worker_fetch_start: float
    worker_respond_with_settled: float
    send_start: float
    send_end: float
    push_start: float
    push_end: float
    receive_headers_start: float
    receive_headers_end: float

    def to_json(self):
        json = dict()
        json['requestTime'] = self.request_time
        json['proxyStart'] = self.proxy_start
        json['proxyEnd'] = self.proxy_end
        json['dnsStart'] = self.dns_start
        json['dnsEnd'] = self.dns_end
        json['connectStart'] = self.connect_start
        json['connectEnd'] = self.connect_end
        json['sslStart'] = self.ssl_start
        json['sslEnd'] = self.ssl_end
        json['workerStart'] = self.worker_start
        json['workerReady'] = self.worker_ready
        json['workerFetchStart'] = self.worker_fetch_start
        json['workerRespondWithSettled'] = self.worker_respond_with_settled
        json['sendStart'] = self.send_start
        json['sendEnd'] = self.send_end
        json['pushStart'] = self.push_start
        json['pushEnd'] = self.push_end
        json['receiveHeadersStart'] = self.receive_headers_start
        json['receiveHeadersEnd'] = self.receive_headers_end
        return json

    @classmethod
    def from_json(cls, json):
        return cls(request_time=float(json['requestTime']), proxy_start=float(json['proxyStart']), proxy_end=float(json['proxyEnd']), dns_start=float(json['dnsStart']), dns_end=float(json['dnsEnd']), connect_start=float(json['connectStart']), connect_end=float(json['connectEnd']), ssl_start=float(json['sslStart']), ssl_end=float(json['sslEnd']), worker_start=float(json['workerStart']), worker_ready=float(json['workerReady']), worker_fetch_start=float(json['workerFetchStart']), worker_respond_with_settled=float(json['workerRespondWithSettled']), send_start=float(json['sendStart']), send_end=float(json['sendEnd']), push_start=float(json['pushStart']), push_end=float(json['pushEnd']), receive_headers_start=float(json['receiveHeadersStart']), receive_headers_end=float(json['receiveHeadersEnd']))