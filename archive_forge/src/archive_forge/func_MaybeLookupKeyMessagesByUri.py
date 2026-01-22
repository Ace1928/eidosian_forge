from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def MaybeLookupKeyMessagesByUri(csek_keys_or_none, parser, uris, compute_client):
    return [MaybeToMessage(k, compute_client) for k in MaybeLookupKeysByUri(csek_keys_or_none, parser, uris)]