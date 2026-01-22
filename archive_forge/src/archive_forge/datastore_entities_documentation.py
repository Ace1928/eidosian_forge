from __future__ import absolute_import
from __future__ import unicode_literals
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.api import datastore
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine._internal import six_subset
 Override GdKind.ToXml() to put some properties inside a
    gd:contactSection.
    