from oslo_serialization import jsonutils
from keystoneclient.generic import client
from keystoneclient.tests.unit import utils
def _create_extension_list(extensions):
    return jsonutils.dumps({'extensions': {'values': extensions}})