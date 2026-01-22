import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _enhance_metadata(self, nodes, content):
    nodemap = {}
    for node in nodes:
        node.extra['vSphere version'] = content.about.version
        nodemap[node.id] = node
    filter_spec = vim.event.EventFilterSpec(eventTypeId=['VmBeingDeployedEvent'])
    deploy_events = content.eventManager.QueryEvent(filter_spec)
    for event in deploy_events:
        try:
            uuid = event.vm.vm.config.instanceUuid
        except Exception:
            continue
        if uuid in nodemap:
            node = nodemap[uuid]
            try:
                source_template_vm = event.srcTemplate.vm
                image_id = source_template_vm.config.instanceUuid
                node.extra['image_id'] = image_id
            except Exception:
                logger.error('Cannot get instanceUuid from source template')
            try:
                node.created_at = event.createdTime
            except AttributeError:
                logger.error('Cannot get creation date from VM deploy event')
    return nodes