from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def GetDefaultAPIProxy():
    try:
        runtime = __import__('googlecloudsdk.third_party.appengine.runtime', globals(), locals(), ['apiproxy'])
        return APIProxyStubMap(runtime.apiproxy)
    except (AttributeError, ImportError):
        return APIProxyStubMap()