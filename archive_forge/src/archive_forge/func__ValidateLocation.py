from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def _ValidateLocation(location):
    if location not in _VALID_LOCATIONS:
        locations = list(_VALID_LOCATIONS)
        locations.sort()
        raise exceptions.InvalidArgumentException('--location', '{bad_location} is not a valid location. Allowed values: [{location_list}].'.format(bad_location=location, location_list=', '.join(("'{}'".format(r) for r in locations))))