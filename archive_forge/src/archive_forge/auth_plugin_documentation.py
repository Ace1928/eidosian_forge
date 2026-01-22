import json
from keystoneauth1 import loading as ks_loading
from oslo_log import log as logging
from heat.common import exception
Validate if this auth_plugin is valid to use.