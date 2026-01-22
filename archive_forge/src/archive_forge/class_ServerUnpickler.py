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
class ServerUnpickler(pickle.Unpickler):

    def persistent_load(self, pid):
        assert isinstance(pid, PickleStub)
        if pid.type == 'Object':
            return ClientObjectRef(pid.ref_id)
        elif pid.type == 'Actor':
            return ClientActorHandle(ClientActorRef(pid.ref_id))
        else:
            raise NotImplementedError('Being passed back an unknown stub')