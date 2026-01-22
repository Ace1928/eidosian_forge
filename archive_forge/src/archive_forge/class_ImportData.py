from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
class ImportData(object):
    """A holder object for yaml import command."""

    def __init__(self, data, orig_request, orig_async):
        self.abort_if_equivalent = data.get('abort_if_equivalent', False)
        self.create_if_not_exists = data.get('create_if_not_exists', False)
        self.no_create_async = data.get('no_create_async', False)
        create_request = data.get('create_request', None)
        if create_request:
            overlayed_create_request = self._OverlayData(create_request, orig_request)
            self.create_request = Request(CommandType.CREATE, overlayed_create_request)
        else:
            self.create_request = None
        create_async = data.get('create_async', None)
        if create_async:
            overlayed_create_async = self._OverlayData(create_async, orig_async)
            self.create_async = Async(overlayed_create_async)
        else:
            self.create_async = None

    def _OverlayData(self, create_data, orig_data):
        """Uses data from the original configuration unless explicitly defined."""
        for k, v in orig_data.items():
            create_data[k] = create_data.get(k) or v
        return create_data