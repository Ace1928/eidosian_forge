from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
def Describe(api_client):
    try:
        return api_client.GetApplication()
    except apitools_exceptions.HttpNotFoundError:
        log.debug('No app found:', exc_info=True)
        project = api_client.project
        raise exceptions.MissingApplicationError(project)