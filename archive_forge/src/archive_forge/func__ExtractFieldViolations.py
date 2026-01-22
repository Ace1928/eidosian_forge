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
def _ExtractFieldViolations(self, details):
    """Extracts a map of field violations from the given error's details.

    Args:
      details: JSON-parsed details field from parsed json of error.

    Returns:
      Map[str, str] field (in dotted format) -> error description.
      The iterator of it is ordered by the order the fields first
      appear in the error.
    """
    results = collections.OrderedDict()
    for deet in details:
        if 'fieldViolations' not in deet:
            continue
        violations = deet['fieldViolations']
        if not isinstance(violations, list):
            continue
        f = deet.get('field')
        for viol in violations:
            try:
                local_f = viol.get('field')
                field = f or local_f
                if field:
                    if field in results:
                        results[field] += '\n' + viol['description']
                    else:
                        results[field] = viol['description']
            except (KeyError, TypeError):
                pass
    return results