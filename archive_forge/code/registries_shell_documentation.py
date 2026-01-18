import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
Update one or more attributes of the registry.