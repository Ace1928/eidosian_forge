from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
def GetDefaultAccountName():
    return MapGaiaEmailToDefaultAccountName(properties.VALUES.core.account.Get())