from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
import sys
import traceback
from googlecloudsdk.third_party.appengine._internal import six_subset
class CronEntry(validation.Validated):
    """A cron entry describes a single cron job."""
    ATTRIBUTES = {URL: _URL_REGEX, SCHEDULE: GrocValidator(), TIMEZONE: validation.Optional(TimezoneValidator()), DESCRIPTION: validation.Optional(_DESCRIPTION_REGEX), RETRY_PARAMETERS: validation.Optional(RetryParameters), TARGET: validation.Optional(_VERSION_REGEX)}