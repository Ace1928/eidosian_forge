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
def _process_child(self, path, request, quiet=True):
    """Receives the result of a child data fetch request."""
    job = None
    try:
        raw_data, node_stat = request.get()
        job_data = misc.decode_json(raw_data)
        job_created_on = misc.millis_to_datetime(node_stat.ctime)
        try:
            job_priority = job_data['priority']
            job_priority = base.JobPriority.convert(job_priority)
        except KeyError:
            job_priority = base.JobPriority.NORMAL
        job_uuid = job_data['uuid']
        job_name = job_data['name']
    except (ValueError, TypeError, KeyError):
        with excutils.save_and_reraise_exception(reraise=not quiet):
            LOG.warning('Incorrectly formatted job data found at path: %s', path, exc_info=True)
    except self._client.handler.timeout_exception:
        with excutils.save_and_reraise_exception(reraise=not quiet):
            LOG.warning('Operation timed out fetching job data from from path: %s', path, exc_info=True)
    except k_exceptions.SessionExpiredError:
        with excutils.save_and_reraise_exception(reraise=not quiet):
            LOG.warning('Session expired fetching job data from path: %s', path, exc_info=True)
    except k_exceptions.NoNodeError:
        LOG.debug('No job node found at path: %s, it must have disappeared or was removed', path)
    except k_exceptions.KazooException:
        with excutils.save_and_reraise_exception(reraise=not quiet):
            LOG.warning('Internal error fetching job data from path: %s', path, exc_info=True)
    else:
        with self._job_cond:
            if path not in self._known_jobs:
                job = ZookeeperJob(self, job_name, self._client, path, backend=self._persistence, uuid=job_uuid, book_data=job_data.get('book'), details=job_data.get('details', {}), created_on=job_created_on, priority=job_priority)
                self._known_jobs[path] = job
                self._job_cond.notify_all()
    if job is not None:
        self._try_emit(base.POSTED, details={'job': job})