import json
import os
import sys
from oslo_utils import strutils
from glanceclient._i18n import _
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import cache
from glanceclient.v2 import image_members
from glanceclient.v2 import image_schema
from glanceclient.v2 import images
from glanceclient.v2 import namespace_schema
from glanceclient.v2 import resource_type_schema
from glanceclient.v2 import tasks
def _namespace_show(namespace, max_column_width=None):
    namespace = dict(namespace)
    if 'properties' in namespace:
        props = [k for k in namespace['properties']]
        namespace['properties'] = props
    if 'resource_type_associations' in namespace:
        assocs = [assoc['name'] for assoc in namespace['resource_type_associations']]
        namespace['resource_type_associations'] = assocs
    if 'objects' in namespace:
        objects = [obj['name'] for obj in namespace['objects']]
        namespace['objects'] = objects
    if 'tags' in namespace:
        tags = [tag['name'] for tag in namespace['tags']]
        namespace['tags'] = tags
    if max_column_width:
        utils.print_dict(namespace, max_column_width)
    else:
        utils.print_dict(namespace)