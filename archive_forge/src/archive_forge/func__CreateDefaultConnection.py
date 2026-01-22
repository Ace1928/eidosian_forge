from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def _CreateDefaultConnection(connection_fn, **kwargs):
    """Creates a new connection to Datastore.

  Uses environment variables to determine if the connection should be made
  to Cloud Datastore v1 or to Datastore's private App Engine API.
  If DATASTORE_PROJECT_ID exists, connect to Cloud Datastore v1. In this case,
  either DATASTORE_APP_ID or DATASTORE_USE_PROJECT_ID_AS_APP_ID must be set to
  indicate what the environment's application should be.

  Args:
    connection_fn: The function to use to create the connection.
    **kwargs: Addition arguments to pass to the connection_fn.

  Raises:
    ValueError: If DATASTORE_PROJECT_ID is set but DATASTORE_APP_ID or
       DATASTORE_USE_PROJECT_ID_AS_APP_ID is not. If DATASTORE_APP_ID doesn't
       resolve to DATASTORE_PROJECT_ID. If DATASTORE_APP_ID doesn't match
       an existing APPLICATION_ID.

  Returns:
    the connection object returned from connection_fn.
  """
    datastore_app_id = os.environ.get(_DATASTORE_APP_ID_ENV, None)
    datastore_project_id = os.environ.get(_DATASTORE_PROJECT_ID_ENV, None)
    if datastore_app_id or datastore_project_id:
        app_id_override = bool(os.environ.get(_DATASTORE_USE_PROJECT_ID_AS_APP_ID_ENV, False))
        if not datastore_app_id and (not app_id_override):
            raise ValueError('Could not determine app id. To use project id (%s) instead, set %s=true. This will affect the serialized form of entities and should not be used if serialized entities will be shared between code running on App Engine and code running off App Engine. Alternatively, set %s=<app id>.' % (datastore_project_id, _DATASTORE_USE_PROJECT_ID_AS_APP_ID_ENV, _DATASTORE_APP_ID_ENV))
        elif datastore_app_id:
            if app_id_override:
                raise ValueError('App id was provided (%s) but %s was set to true. Please unset either %s or %s.' % (datastore_app_id, _DATASTORE_USE_PROJECT_ID_AS_APP_ID_ENV, _DATASTORE_APP_ID_ENV, _DATASTORE_USE_PROJECT_ID_AS_APP_ID_ENV))
            elif datastore_project_id:
                id_resolver = datastore_pbs.IdResolver([datastore_app_id])
                if datastore_project_id != id_resolver.resolve_project_id(datastore_app_id):
                    raise ValueError('App id "%s" does not match project id "%s".' % (datastore_app_id, datastore_project_id))
        datastore_app_id = datastore_app_id or datastore_project_id
        additional_app_str = os.environ.get(_DATASTORE_ADDITIONAL_APP_IDS_ENV, '')
        additional_apps = (app.strip() for app in additional_app_str.split(','))
        return _CreateCloudDatastoreConnection(connection_fn, datastore_app_id, additional_apps, kwargs)
    return connection_fn(**kwargs)