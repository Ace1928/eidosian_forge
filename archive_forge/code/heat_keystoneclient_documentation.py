import collections
import uuid
import weakref
from keystoneauth1 import exceptions as ks_exception
from keystoneauth1.identity import generic as ks_auth
from keystoneclient.v3 import client as kc_v3
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import importutils
from heat.common import config
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
Get an ec2 keypair via v3/credentials, by id or access.