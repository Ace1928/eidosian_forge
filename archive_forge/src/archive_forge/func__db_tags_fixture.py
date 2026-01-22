import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def _db_tags_fixture(tag_names=None):
    tag_list = []
    if not tag_names:
        tag_names = [TAG1, TAG2, TAG3]
    for tag_name in tag_names:
        tag = tags.MetadefTag()
        tag.name = tag_name
        tag_list.append(tag)
    return tag_list