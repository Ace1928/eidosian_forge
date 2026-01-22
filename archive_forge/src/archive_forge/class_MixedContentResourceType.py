from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
class MixedContentResourceType(enum.Enum):
    AUDIO = 'Audio'
    BEACON = 'Beacon'
    CSP_REPORT = 'CSPReport'
    DOWNLOAD = 'Download'
    EVENT_SOURCE = 'EventSource'
    FAVICON = 'Favicon'
    FONT = 'Font'
    FORM = 'Form'
    FRAME = 'Frame'
    IMAGE = 'Image'
    IMPORT = 'Import'
    MANIFEST = 'Manifest'
    PING = 'Ping'
    PLUGIN_DATA = 'PluginData'
    PLUGIN_RESOURCE = 'PluginResource'
    PREFETCH = 'Prefetch'
    RESOURCE = 'Resource'
    SCRIPT = 'Script'
    SERVICE_WORKER = 'ServiceWorker'
    SHARED_WORKER = 'SharedWorker'
    STYLESHEET = 'Stylesheet'
    TRACK = 'Track'
    VIDEO = 'Video'
    WORKER = 'Worker'
    XML_HTTP_REQUEST = 'XMLHttpRequest'
    XSLT = 'XSLT'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)