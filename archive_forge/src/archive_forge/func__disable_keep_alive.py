import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def _disable_keep_alive(conn):
    sess = conn.config.get_session()
    sess.keep_alive = False