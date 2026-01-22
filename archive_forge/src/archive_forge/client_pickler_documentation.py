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
Implements the client side of the client/server pickling protocol.

All ray client client/server data transfer happens through this pickling
protocol. The model is as follows:

    * All Client objects (eg ClientObjectRef) always live on the client and
      are never represented in the server
    * All Ray objects (eg, ray.ObjectRef) always live on the server and are
      never returned to the client
    * In order to translate between these two references, PickleStub tuples
      are generated as persistent ids in the data blobs during the pickling
      and unpickling of these objects.

The PickleStubs have just enough information to find or generate their
associated partner object on either side.

This also has the advantage of avoiding predefined pickle behavior for ray
objects, which may include ray internal reference counting.

ClientPickler dumps things from the client into the appropriate stubs
ServerUnpickler loads stubs from the server into their client counterparts.
