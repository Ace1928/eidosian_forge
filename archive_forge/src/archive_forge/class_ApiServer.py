import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
class ApiServer(Server):
    """
    Server object that starts/stops/manages the API server
    """

    def __init__(self, test_dir, port, policy_file, delayed_delete=False, pid_file=None, sock=None, **kwargs):
        super(ApiServer, self).__init__(test_dir, port, sock=sock)
        self.server_name = 'api'
        self.server_module = 'glance.cmd.%s' % self.server_name
        self.default_store = kwargs.get('default_store', 'file')
        self.bind_host = '127.0.0.1'
        self.metadata_encryption_key = '012345678901234567890123456789ab'
        self.image_dir = os.path.join(self.test_dir, 'images')
        self.pid_file = pid_file or os.path.join(self.test_dir, 'api.pid')
        self.log_file = os.path.join(self.test_dir, 'api.log')
        self.image_size_cap = 1099511627776
        self.delayed_delete = delayed_delete
        self.workers = 0
        self.scrub_time = 5
        self.image_cache_dir = os.path.join(self.test_dir, 'cache')
        self.image_cache_driver = 'sqlite'
        self.policy_file = policy_file
        self.policy_default_rule = 'default'
        self.property_protection_rule_format = 'roles'
        self.image_member_quota = 10
        self.image_property_quota = 10
        self.image_tag_quota = 10
        self.image_location_quota = 2
        self.disable_path = None
        self.enforce_new_defaults = True
        self.needs_database = True
        default_sql_connection = SQLITE_CONN_TEMPLATE % self.test_dir
        self.sql_connection = os.environ.get('GLANCE_TEST_SQL_CONNECTION', default_sql_connection)
        self.user_storage_quota = '0'
        self.lock_path = self.test_dir
        self.location_strategy = 'location_order'
        self.store_type_location_strategy_preference = ''
        self.node_staging_uri = 'file://%s' % os.path.join(self.test_dir, 'staging')
        self.conf_base = '[DEFAULT]\ndebug = %(debug)s\ndefault_log_levels = eventlet.wsgi.server=DEBUG,stevedore.extension=INFO\nbind_host = %(bind_host)s\nbind_port = %(bind_port)s\nmetadata_encryption_key = %(metadata_encryption_key)s\nlog_file = %(log_file)s\nimage_size_cap = %(image_size_cap)d\ndelayed_delete = %(delayed_delete)s\nworkers = %(workers)s\nscrub_time = %(scrub_time)s\nimage_cache_dir = %(image_cache_dir)s\nimage_cache_driver = %(image_cache_driver)s\nshow_image_direct_url = %(show_image_direct_url)s\nshow_multiple_locations = %(show_multiple_locations)s\nuser_storage_quota = %(user_storage_quota)s\nlock_path = %(lock_path)s\nproperty_protection_file = %(property_protection_file)s\nproperty_protection_rule_format = %(property_protection_rule_format)s\nimage_member_quota=%(image_member_quota)s\nimage_property_quota=%(image_property_quota)s\nimage_tag_quota=%(image_tag_quota)s\nimage_location_quota=%(image_location_quota)s\nlocation_strategy=%(location_strategy)s\nallow_additional_image_properties = True\nnode_staging_uri=%(node_staging_uri)s\n[database]\nconnection = %(sql_connection)s\n[oslo_policy]\npolicy_file = %(policy_file)s\npolicy_default_rule = %(policy_default_rule)s\nenforce_new_defaults=%(enforce_new_defaults)s\n[paste_deploy]\nflavor = %(deployment_flavor)s\n[store_type_location_strategy]\nstore_type_preference = %(store_type_location_strategy_preference)s\n[glance_store]\nfilesystem_store_datadir=%(image_dir)s\ndefault_store = %(default_store)s\n[import_filtering_opts]\nallowed_ports = []\n'
        self.paste_conf_base = '[composite:glance-api]\npaste.composite_factory = glance.api:root_app_factory\n/: api\n/healthcheck: healthcheck\n\n[pipeline:api]\npipeline =\n    cors\n    versionnegotiation\n    gzip\n    unauthenticated-context\n    rootapp\n\n[composite:glance-api-caching]\npaste.composite_factory = glance.api:root_app_factory\n/: api-caching\n/healthcheck: healthcheck\n\n[pipeline:api-caching]\npipeline = cors versionnegotiation gzip context cache rootapp\n\n[composite:glance-api-cachemanagement]\npaste.composite_factory = glance.api:root_app_factory\n/: api-cachemanagement\n/healthcheck: healthcheck\n\n[pipeline:api-cachemanagement]\npipeline =\n    cors\n    versionnegotiation\n    gzip\n    unauthenticated-context\n    cache\n    cache_manage\n    rootapp\n\n[composite:glance-api-fakeauth]\npaste.composite_factory = glance.api:root_app_factory\n/: api-fakeauth\n/healthcheck: healthcheck\n\n[pipeline:api-fakeauth]\npipeline = cors versionnegotiation gzip fakeauth context rootapp\n\n[composite:glance-api-noauth]\npaste.composite_factory = glance.api:root_app_factory\n/: api-noauth\n/healthcheck: healthcheck\n\n[pipeline:api-noauth]\npipeline = cors versionnegotiation gzip context rootapp\n\n[composite:rootapp]\npaste.composite_factory = glance.api:root_app_factory\n/: apiversions\n/v2: apiv2app\n\n[app:apiversions]\npaste.app_factory = glance.api.versions:create_resource\n\n[app:apiv2app]\npaste.app_factory = glance.api.v2.router:API.factory\n\n[app:healthcheck]\npaste.app_factory = oslo_middleware:Healthcheck.app_factory\nbackends = disable_by_file\ndisable_by_file_path = %(disable_path)s\n\n[filter:versionnegotiation]\npaste.filter_factory = glance.api.middleware.version_negotiation:VersionNegotiationFilter.factory\n\n[filter:gzip]\npaste.filter_factory = glance.api.middleware.gzip:GzipMiddleware.factory\n\n[filter:cache]\npaste.filter_factory = glance.api.middleware.cache:CacheFilter.factory\n\n[filter:cache_manage]\npaste.filter_factory = glance.api.middleware.cache_manage:CacheManageFilter.factory\n\n[filter:context]\npaste.filter_factory = glance.api.middleware.context:ContextMiddleware.factory\n\n[filter:unauthenticated-context]\npaste.filter_factory = glance.api.middleware.context:UnauthenticatedContextMiddleware.factory\n\n[filter:fakeauth]\npaste.filter_factory = glance.tests.utils:FakeAuthMiddleware.factory\n\n[filter:cors]\npaste.filter_factory = oslo_middleware.cors:filter_factory\nallowed_origin=http://valid.example.com\n'