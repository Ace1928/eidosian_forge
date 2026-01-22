from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def IsDefaultRegionalBackendServicePropertyNoneWarnOtherwise():
    """Warns if core/default_regional_backend_service property is set."""
    default_regional = properties.VALUES.core.default_regional_backend_service.GetBool()
    if default_regional is not None:
        log.warning('core/default_regional_backend_service property is deprecated and has no meaning.')