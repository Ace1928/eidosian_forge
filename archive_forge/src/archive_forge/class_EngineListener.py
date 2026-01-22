import collections
import copy
import datetime
import functools
import itertools
import os
import pydoc
import signal
import socket
import sys
import eventlet
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_serialization import jsonutils
from oslo_service import service
from oslo_service import threadgroup
from oslo_utils import timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
import webob
from heat.common import context
from heat.common import environment_format as env_fmt
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import messaging as rpc_messaging
from heat.common import policy
from heat.common import service_utils
from heat.engine import api
from heat.engine import attributes
from heat.engine.cfn import template as cfntemplate
from heat.engine import clients
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine import parameter_groups
from heat.engine import properties
from heat.engine import resources
from heat.engine import service_software_config
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.engine import stk_defn
from heat.engine import support
from heat.engine import template as templatem
from heat.engine import template_files
from heat.engine import update
from heat.engine import worker
from heat.objects import event as event_object
from heat.objects import resource as resource_objects
from heat.objects import service as service_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_api as rpc_worker_api
@profiler.trace_cls('rpc')
class EngineListener(object):
    """Listen on an AMQP queue named for the engine.

    Allows individual engines to communicate with each other for multi-engine
    support.
    """
    ACTIONS = STOP_STACK, SEND = ('stop_stack', 'send')

    def __init__(self, host, engine_id, thread_group_mgr):
        self.thread_group_mgr = thread_group_mgr
        self.engine_id = engine_id
        self.host = host
        self._server = None

    def start(self):
        self.target = messaging.Target(server=self.engine_id, topic=rpc_api.LISTENER_TOPIC)
        self._server = rpc_messaging.get_rpc_server(self.target, self)
        self._server.start()

    def stop(self):
        if self._server is not None:
            LOG.debug('Attempting to stop engine listener...')
            try:
                self._server.stop()
                self._server.wait()
                LOG.info('Engine listener is stopped successfully')
            except Exception as e:
                LOG.error('Failed to stop engine listener, %s', e)

    def listening(self, ctxt):
        """Respond to a watchdog request.

        Respond affirmatively to confirm that the engine performing the action
        is still alive.
        """
        return True

    def stop_stack(self, ctxt, stack_identity):
        """Stop any active threads on a stack."""
        stack_id = stack_identity['stack_id']
        self.thread_group_mgr.stop(stack_id)

    def send(self, ctxt, stack_identity, message):
        stack_id = stack_identity['stack_id']
        self.thread_group_mgr.send(stack_id, message)