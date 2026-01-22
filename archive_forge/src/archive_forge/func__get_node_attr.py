import collections
import contextlib
import functools
import sys
import threading
import fasteners
import futurist
from kazoo import exceptions as k_exceptions
from kazoo.protocol import paths as k_paths
from kazoo.protocol import states as k_states
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow.conductors import base as c_base
from taskflow import exceptions as excp
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
def _get_node_attr(self, path, attr_name, trans_func=None):
    try:
        _data, node_stat = self._client.get(path)
        attr = getattr(node_stat, attr_name)
        if trans_func is not None:
            return trans_func(attr)
        else:
            return attr
    except k_exceptions.NoNodeError:
        excp.raise_with_cause(excp.NotFound, 'Can not fetch the %r attribute of job %s (%s), path %s not found' % (attr_name, self.uuid, self.path, path))
    except self._client.handler.timeout_exception:
        excp.raise_with_cause(excp.JobFailure, 'Can not fetch the %r attribute of job %s (%s), operation timed out' % (attr_name, self.uuid, self.path))
    except k_exceptions.SessionExpiredError:
        excp.raise_with_cause(excp.JobFailure, 'Can not fetch the %r attribute of job %s (%s), session expired' % (attr_name, self.uuid, self.path))
    except (AttributeError, k_exceptions.KazooException):
        excp.raise_with_cause(excp.JobFailure, 'Can not fetch the %r attribute of job %s (%s), internal error' % (attr_name, self.uuid, self.path))