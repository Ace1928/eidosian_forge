import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
@classmethod
def wrap_collection(cls, refs, hints=None, collection_name=None):
    """Wrap a collection, checking for filtering and pagination.

        Returns the wrapped collection, which includes:
        - Executing any filtering not already carried out
        - Truncate to a set limit if necessary
        - Adds 'self' links in every member
        - Adds 'next', 'self' and 'prev' links for the whole collection.

        :param refs: the list of members of the collection
        :param hints: list hints, containing any relevant filters and limit.
                      Any filters already satisfied by managers will have been
                      removed
        :param collection_name: optional override for the 'collection key'
                                class attribute. This is to be used when
                                wrapping a collection for a different api,
                                e.g. 'roles' from the 'trust' api.
        """
    if hints:
        refs = cls.filter_by_attributes(refs, hints)
    list_limited, refs = cls.limit(refs, hints)
    collection = collection_name or cls.collection_key
    for ref in refs:
        cls._add_self_referential_link(ref, collection_name=collection)
    container = {collection: refs}
    self_url = full_url(flask.request.environ['PATH_INFO'])
    container['links'] = {'next': None, 'self': self_url, 'previous': None}
    if list_limited:
        container['truncated'] = True
    return container