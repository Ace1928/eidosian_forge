from oslo_utils import encodeutils
from urllib import parse
from heatclient.common import base
from heatclient.common import utils
Get the details for a specific resource_type.

        :param resource_type: name of the resource type to get the details for
        :param with_description: return result with description or not
        