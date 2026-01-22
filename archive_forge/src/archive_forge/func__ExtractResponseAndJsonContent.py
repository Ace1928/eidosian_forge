from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def _ExtractResponseAndJsonContent(self, http_error):
    """Extracts the response and JSON content from the HttpError."""
    response = getattr(http_error, 'response', None)
    if response:
        self.status_code = int(response.get('status', 0))
        self.status_description = encoding.Decode(response.get('reason', ''))
    content = encoding.Decode(http_error.content)
    try:
        self.content = _JsonSortedDict(json.loads(content))
        self.error_info = _JsonSortedDict(self.content['error'])
        if not self.status_code:
            self.status_code = int(self.error_info.get('code', 0))
        if not self.status_description:
            self.status_description = self.error_info.get('status', '')
        self.status_message = self.error_info.get('message', '')
        self.details = self.error_info.get('details', [])
        self.violations = self._ExtractViolations(self.details)
        self.field_violations = self._ExtractFieldViolations(self.details)
        self.type_details = self._IndexErrorDetailsByType(self.details)
        self.domain_details = self._IndexErrorInfoByDomain(self.details)
    except (KeyError, TypeError, ValueError):
        self.status_message = content
    except AttributeError:
        pass