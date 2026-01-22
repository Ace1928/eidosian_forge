from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Type
from simpy.core import BoundClass, Environment, SimTime
from simpy.resources import base
class PreemptiveResource(PriorityResource):
    """A :class:`~simpy.resources.resource.PriorityResource` with preemption.

    If a request is preempted, the process of that request will receive an
    :class:`~simpy.exceptions.Interrupt` with a :class:`Preempted` instance as
    cause.

    """
    users: List[PriorityRequest]

    def _do_put(self, event: PriorityRequest) -> None:
        if len(self.users) >= self.capacity and event.preempt:
            preempt = sorted(self.users, key=lambda e: e.key)[-1]
            if preempt.key > event.key:
                self.users.remove(preempt)
                preempt.proc.interrupt(Preempted(by=event.proc, usage_since=preempt.usage_since, resource=self))
        return super()._do_put(event)