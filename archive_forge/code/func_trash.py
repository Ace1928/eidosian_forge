import contextlib
import datetime
import functools
import re
import string
import threading
import time
import fasteners
import msgpack
from oslo_serialization import msgpackutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from redis import exceptions as redis_exceptions
from redis import sentinel
from taskflow import exceptions as exc
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import misc
from taskflow.utils import redis_utils as ru
@base.check_who
def trash(self, job, who):
    script = self._get_script('trash')
    with _translate_failures():
        raw_who = self._encode_owner(who)
        raw_result = script(keys=[job.owner_key, self.listings_key, job.last_modified_key, self.trash_key], args=[raw_who, job.key, self._dumps(timeutils.utcnow())])
        result = self._loads(raw_result)
    status = result['status']
    if status != self.SCRIPT_STATUS_OK:
        reason = result.get('reason')
        if reason == self.SCRIPT_UNKNOWN_JOB:
            raise exc.NotFound('Job %s not found to be trashed' % job.uuid)
        elif reason == self.SCRIPT_UNKNOWN_OWNER:
            raise exc.NotFound('Can not trash job %s which we can not determine the owner of' % job.uuid)
        elif reason == self.SCRIPT_NOT_EXPECTED_OWNER:
            raw_owner = result.get('owner')
            if raw_owner:
                owner = self._decode_owner(raw_owner)
                raise exc.JobFailure('Can not trash job %s which is not owned by %s (it is actively owned by %s)' % (job.uuid, who, owner))
            else:
                raise exc.JobFailure('Can not trash job %s which is not owned by %s' % (job.uuid, who))
        else:
            raise exc.JobFailure('Failure to trash job %s, unknown internal error (reason=%s)' % (job.uuid, reason))