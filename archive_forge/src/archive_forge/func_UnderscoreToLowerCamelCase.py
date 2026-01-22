from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
def UnderscoreToLowerCamelCase(text):
    """Convert underscores to lower camel case (e.g. 'foo_bar' --> 'fooBar')."""
    parts = text.lower().split('_')
    return parts[0] + ''.join((part.capitalize() for part in parts[1:]))