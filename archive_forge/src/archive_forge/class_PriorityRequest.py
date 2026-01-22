from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Type
from simpy.core import BoundClass, Environment, SimTime
from simpy.resources import base
class PriorityRequest(Request):
    """Request the usage of *resource* with a given *priority*. If the
    *resource* supports preemption and *preempt* is ``True`` other usage
    requests of the *resource* may be preempted (see
    :class:`PreemptiveResource` for details).

    This event type inherits :class:`Request` and adds some additional
    attributes needed by :class:`PriorityResource` and
    :class:`PreemptiveResource`

    """

    def __init__(self, resource: Resource, priority: int=0, preempt: bool=True):
        self.priority = priority
        'The priority of this request. A smaller number means higher\n        priority.'
        self.preempt = preempt
        'Indicates whether the request should preempt a resource user or not\n        (:class:`PriorityResource` ignores this flag).'
        self.time = resource._env.now
        'The time at which the request was made.'
        self.key = (self.priority, self.time, not self.preempt)
        'Key for sorting events. Consists of the priority (lower value is\n        more important), the time at which the request was made (earlier\n        requests are more important) and finally the preemption flag (preempt\n        requests are more important).'
        super().__init__(resource)