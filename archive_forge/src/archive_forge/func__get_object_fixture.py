import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def _get_object_fixture(ns_name, obj_name, **kwargs):
    obj = {'description': 'DESCRIPTION', 'name': obj_name, 'self': '/v2/metadefs/namespaces/%s/objects/%s' % (ns_name, obj_name), 'required': [], 'properties': {PROPERTY1: {'type': 'integer', 'description': 'DESCRIPTION', 'title': 'Quota: CPU Shares'}, PROPERTY2: {'minimum': 1000, 'type': 'integer', 'description': 'DESCRIPTION', 'maximum': 1000000, 'title': 'Quota: CPU Period'}}, 'schema': '/v2/schemas/metadefs/object', 'created_at': '2014-08-14T09:07:06Z', 'updated_at': '2014-08-14T09:07:06Z'}
    obj.update(kwargs)
    return obj