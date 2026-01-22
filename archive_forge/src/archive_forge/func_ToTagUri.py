from the atom and gd namespaces. For more information, see:
from __future__ import absolute_import
import base64
import calendar
import datetime
import os
import re
import time
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import namespace_manager
from googlecloudsdk.third_party.appengine.api import users
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
from googlecloudsdk.third_party.appengine.datastore import sortable_pb_encoder
from googlecloudsdk.third_party.appengine._internal import six_subset
def ToTagUri(self):
    """Returns a tag: URI for this entity for use in XML output.

    Foreign keys for entities may be represented in XML output as tag URIs.
    RFC 4151 describes the tag URI scheme. From http://taguri.org/:

      The tag algorithm lets people mint - create - identifiers that no one
      else using the same algorithm could ever mint. It is simple enough to do
      in your head, and the resulting identifiers can be easy to read, write,
      and remember. The identifiers conform to the URI (URL) Syntax.

    Tag URIs for entities use the app's auth domain and the date that the URI
     is generated. The namespace-specific part is <kind>[<key>].

    For example, here is the tag URI for a Kitten with the key "Fluffy" in the
    catsinsinks app:

      tag:catsinsinks.googleapps.com,2006-08-29:Kitten[Fluffy]

    Raises a BadKeyError if this entity's key is incomplete.
    """
    if not self.has_id_or_name():
        raise datastore_errors.BadKeyError('ToTagUri() called for an entity with an incomplete key.')
    return u'tag:%s.%s,%s:%s[%s]' % (saxutils.escape(EncodeAppIdNamespace(self.app(), self.namespace())), os.environ['AUTH_DOMAIN'], datetime.date.today().isoformat(), saxutils.escape(self.kind()), saxutils.escape(str(self)))