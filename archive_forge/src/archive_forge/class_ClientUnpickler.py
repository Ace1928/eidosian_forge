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
class ClientUnpickler(pickle.Unpickler):

    def __init__(self, server, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = server

    def persistent_load(self, pid):
        assert isinstance(pid, PickleStub)
        if pid.type == 'Ray':
            return ray
        elif pid.type == 'Object':
            return self.server.object_refs[pid.client_id][pid.ref_id]
        elif pid.type == 'Actor':
            return self.server.actor_refs[pid.ref_id]
        elif pid.type == 'RemoteFuncSelfReference':
            return ClientReferenceFunction(pid.client_id, pid.ref_id)
        elif pid.type == 'RemoteFunc':
            return self.server.lookup_or_register_func(pid.ref_id, pid.client_id, pid.baseline_options)
        elif pid.type == 'RemoteActorSelfReference':
            return ClientReferenceActor(pid.client_id, pid.ref_id)
        elif pid.type == 'RemoteActor':
            return self.server.lookup_or_register_actor(pid.ref_id, pid.client_id, pid.baseline_options)
        elif pid.type == 'RemoteMethod':
            actor = self.server.actor_refs[pid.ref_id]
            return getattr(actor, pid.name)
        else:
            raise NotImplementedError('Uncovered client data type')