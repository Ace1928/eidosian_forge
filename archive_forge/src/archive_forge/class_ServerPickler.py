import io
import ray
from typing import Any
from typing import TYPE_CHECKING
from ray._private.client_mode_hook import disable_client_hook
import ray.cloudpickle as cloudpickle
from ray.util.client.client_pickler import PickleStub
from ray.util.client.server.server_stubs import ClientReferenceActor
from ray.util.client.server.server_stubs import ClientReferenceFunction
import pickle  # noqa: F401
class ServerPickler(cloudpickle.CloudPickler):

    def __init__(self, client_id: str, server: 'RayletServicer', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.server = server

    def persistent_id(self, obj):
        if isinstance(obj, ray.ObjectRef):
            obj_id = obj.binary()
            if obj_id not in self.server.object_refs[self.client_id]:
                self.server.object_refs[self.client_id][obj_id] = obj
            return PickleStub(type='Object', client_id=self.client_id, ref_id=obj_id, name=None, baseline_options=None)
        elif isinstance(obj, ray.actor.ActorHandle):
            actor_id = obj._actor_id.binary()
            if actor_id not in self.server.actor_refs:
                self.server.actor_refs[actor_id] = obj
            if actor_id not in self.server.actor_owners[self.client_id]:
                self.server.actor_owners[self.client_id].add(actor_id)
            return PickleStub(type='Actor', client_id=self.client_id, ref_id=obj._actor_id.binary(), name=None, baseline_options=None)
        return None