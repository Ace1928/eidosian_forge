from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _TargetProxySslCertificatesToCell(target_proxy):
    """Joins the names of ssl certificates of the given HTTPS or SSL proxy."""
    return ','.join((path_simplifier.Name(cert) for cert in target_proxy.get('sslCertificates', [])))