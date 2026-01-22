import numbers
from collections import namedtuple
from collections.abc import Mapping
from datetime import timedelta
from weakref import WeakValueDictionary
from kombu import Connection, Consumer, Exchange, Producer, Queue, pools
from kombu.common import Broadcast
from kombu.utils.functional import maybe_list
from kombu.utils.objects import cached_property
from celery import signals
from celery.utils.nodenames import anon_nodename
from celery.utils.saferepr import saferepr
from celery.utils.text import indent as textindent
from celery.utils.time import maybe_make_aware
from . import routes as _routes
def as_task_v2(self, task_id, name, args=None, kwargs=None, countdown=None, eta=None, group_id=None, group_index=None, expires=None, retries=0, chord=None, callbacks=None, errbacks=None, reply_to=None, time_limit=None, soft_time_limit=None, create_sent_event=False, root_id=None, parent_id=None, shadow=None, chain=None, now=None, timezone=None, origin=None, ignore_result=False, argsrepr=None, kwargsrepr=None, stamped_headers=None, replaced_task_nesting=0, **options):
    args = args or ()
    kwargs = kwargs or {}
    if not isinstance(args, (list, tuple)):
        raise TypeError('task args must be a list or tuple')
    if not isinstance(kwargs, Mapping):
        raise TypeError('task keyword arguments must be a mapping')
    if countdown:
        self._verify_seconds(countdown, 'countdown')
        now = now or self.app.now()
        timezone = timezone or self.app.timezone
        eta = maybe_make_aware(now + timedelta(seconds=countdown), tz=timezone)
    if isinstance(expires, numbers.Real):
        self._verify_seconds(expires, 'expires')
        now = now or self.app.now()
        timezone = timezone or self.app.timezone
        expires = maybe_make_aware(now + timedelta(seconds=expires), tz=timezone)
    if not isinstance(eta, str):
        eta = eta and eta.isoformat()
    if not isinstance(expires, str):
        expires = expires and expires.isoformat()
    if argsrepr is None:
        argsrepr = saferepr(args, self.argsrepr_maxsize)
    if kwargsrepr is None:
        kwargsrepr = saferepr(kwargs, self.kwargsrepr_maxsize)
    if not root_id:
        root_id = task_id
    stamps = {header: options[header] for header in stamped_headers or []}
    headers = {'lang': 'py', 'task': name, 'id': task_id, 'shadow': shadow, 'eta': eta, 'expires': expires, 'group': group_id, 'group_index': group_index, 'retries': retries, 'timelimit': [time_limit, soft_time_limit], 'root_id': root_id, 'parent_id': parent_id, 'argsrepr': argsrepr, 'kwargsrepr': kwargsrepr, 'origin': origin or anon_nodename(), 'ignore_result': ignore_result, 'replaced_task_nesting': replaced_task_nesting, 'stamped_headers': stamped_headers, 'stamps': stamps}
    return task_message(headers=headers, properties={'correlation_id': task_id, 'reply_to': reply_to or ''}, body=(args, kwargs, {'callbacks': callbacks, 'errbacks': errbacks, 'chain': chain, 'chord': chord}), sent_event={'uuid': task_id, 'root_id': root_id, 'parent_id': parent_id, 'name': name, 'args': argsrepr, 'kwargs': kwargsrepr, 'retries': retries, 'eta': eta, 'expires': expires} if create_sent_event else None)