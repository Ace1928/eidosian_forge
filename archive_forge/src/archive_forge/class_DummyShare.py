import collections
import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share as mshare
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class DummyShare(object):

    def __init__(self):
        self.availability_zone = 'az'
        self.host = 'host'
        self.share_server_id = 'id'
        self.created_at = 'ca'
        self.status = 's'
        self.project_id = 'p_id'