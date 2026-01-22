from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def GetHandlerType(self):
    """Gets the handler type of a mapping.

    Returns:
      The handler type as determined by which handler ID attribute is set.

    Raises:
      UnknownHandlerType: If none of the handler ID attributes are set.
      UnexpectedHandlerAttribute: If an unexpected attribute is set for the
          discovered handler type.
      HandlerTypeMissingAttribute: If the handler is missing a required
          attribute for its handler type.
      MissingHandlerAttribute: If a URL handler is missing an attribute.
    """
    if getattr(self, HANDLER_API_ENDPOINT) is not None:
        mapping_type = HANDLER_API_ENDPOINT
    else:
        for id_field in URLMap.ALLOWED_FIELDS:
            if getattr(self, id_field) is not None:
                mapping_type = id_field
                break
        else:
            raise appinfo_errors.UnknownHandlerType('Unknown url handler type.\n%s' % str(self))
    allowed_fields = URLMap.ALLOWED_FIELDS[mapping_type]
    for attribute in self.ATTRIBUTES:
        if getattr(self, attribute) is not None and (not (attribute in allowed_fields or attribute in URLMap.COMMON_FIELDS or attribute == mapping_type)):
            raise appinfo_errors.UnexpectedHandlerAttribute('Unexpected attribute "%s" for mapping type %s.' % (attribute, mapping_type))
    if mapping_type == HANDLER_STATIC_FILES and (not self.upload):
        raise appinfo_errors.MissingHandlerAttribute('Missing "%s" attribute for URL "%s".' % (UPLOAD, self.url))
    return mapping_type