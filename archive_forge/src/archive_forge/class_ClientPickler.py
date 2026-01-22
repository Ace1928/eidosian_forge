import io
from typing import NamedTuple
from typing import Any
from typing import Dict
from typing import Optional
import ray.cloudpickle as cloudpickle
from ray.util.client import RayAPIStub
from ray.util.client.common import ClientObjectRef
from ray.util.client.common import ClientActorHandle
from ray.util.client.common import ClientActorRef
from ray.util.client.common import ClientActorClass
from ray.util.client.common import ClientRemoteFunc
from ray.util.client.common import ClientRemoteMethod
from ray.util.client.common import OptionWrapper
from ray.util.client.common import InProgressSentinel
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import pickle  # noqa: F401
class ClientPickler(cloudpickle.CloudPickler):

    def __init__(self, client_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id

    def persistent_id(self, obj):
        if isinstance(obj, RayAPIStub):
            return PickleStub(type='Ray', client_id=self.client_id, ref_id=b'', name=None, baseline_options=None)
        elif isinstance(obj, ClientObjectRef):
            return PickleStub(type='Object', client_id=self.client_id, ref_id=obj.id, name=None, baseline_options=None)
        elif isinstance(obj, ClientActorHandle):
            return PickleStub(type='Actor', client_id=self.client_id, ref_id=obj._actor_id.id, name=None, baseline_options=None)
        elif isinstance(obj, ClientRemoteFunc):
            if obj._ref is None:
                obj._ensure_ref()
            if type(obj._ref) == InProgressSentinel:
                return PickleStub(type='RemoteFuncSelfReference', client_id=self.client_id, ref_id=obj._client_side_ref.id, name=None, baseline_options=None)
            return PickleStub(type='RemoteFunc', client_id=self.client_id, ref_id=obj._ref.id, name=None, baseline_options=obj._options)
        elif isinstance(obj, ClientActorClass):
            if obj._ref is None:
                obj._ensure_ref()
            if type(obj._ref) == InProgressSentinel:
                return PickleStub(type='RemoteActorSelfReference', client_id=self.client_id, ref_id=obj._client_side_ref.id, name=None, baseline_options=None)
            return PickleStub(type='RemoteActor', client_id=self.client_id, ref_id=obj._ref.id, name=None, baseline_options=obj._options)
        elif isinstance(obj, ClientRemoteMethod):
            return PickleStub(type='RemoteMethod', client_id=self.client_id, ref_id=obj._actor_handle.actor_ref.id, name=obj._method_name, baseline_options=None)
        elif isinstance(obj, OptionWrapper):
            raise NotImplementedError('Sending a partial option is unimplemented')
        return None