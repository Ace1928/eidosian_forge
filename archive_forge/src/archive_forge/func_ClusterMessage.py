from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.core.console import console_io
def ClusterMessage(name, action=None, kind=None, location=None):
    msg = 'cluster [{name}]'.format(name=name)
    if action:
        msg = '{action} '.format(action=action) + msg
    if kind and location:
        msg += ' in {kind} location [{location}]'.format(kind=kind, location=location)
    return msg