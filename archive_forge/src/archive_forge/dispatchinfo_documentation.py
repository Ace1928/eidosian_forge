from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
Initializes this ParsedURL with an URL pattern value.

    Args:
      url_pattern: An URL pattern that conforms to the regular expression
          '^([^/]+)(/.*)$'.

    Raises:
      validation.ValidationError: When url_pattern does not match the required
          regular expression.
    