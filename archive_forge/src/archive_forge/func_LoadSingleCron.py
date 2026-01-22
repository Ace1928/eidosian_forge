from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
import sys
import traceback
from googlecloudsdk.third_party.appengine._internal import six_subset
def LoadSingleCron(cron_info, open_fn=None):
    """Load a cron.yaml file or string and return a CronInfoExternal object."""
    builder = yaml_object.ObjectBuilder(CronInfoExternal)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(cron_info)
    cron_info_result = handler.GetResults()
    if len(cron_info_result) < 1:
        raise MalformedCronfigurationFile('Empty cron configuration.')
    if len(cron_info_result) > 1:
        raise MalformedCronfigurationFile('Multiple cron sections in configuration.')
    return cron_info_result[0]