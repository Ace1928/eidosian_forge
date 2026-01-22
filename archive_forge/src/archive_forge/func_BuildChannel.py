from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc import common_publishing
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def BuildChannel(self, channel_ref, provider_ref, crypto_key_name):
    return self._messages.Channel(name=channel_ref.RelativeName(), cryptoKeyName=crypto_key_name, provider=provider_ref if provider_ref is None else provider_ref.RelativeName())