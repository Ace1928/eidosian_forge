import ray
import ray._private.profiling as profiling
import ray._private.services as services
import ray._private.utils as utils
import ray._private.worker
from ray._private import ray_constants
from ray._private.state import GlobalState
from ray._raylet import GcsClientOptions
Free a list of IDs from the in-process and plasma object stores.

    This function is a low-level API which should be used in restricted
    scenarios.

    If local_only is false, the request will be send to all object stores.

    This method will not return any value to indicate whether the deletion is
    successful or not. This function is an instruction to the object store. If
    some of the objects are in use, the object stores will delete them later
    when the ref count is down to 0.

    Examples:

        .. testcode::

            import ray

            @ray.remote
            def f():
                return 0

            obj_ref = f.remote()
            ray.get(obj_ref)  # wait for object to be created first
            free([obj_ref])  # unpin & delete object globally

    Args:
        object_refs (List[ObjectRef]): List of object refs to delete.
        local_only: Whether only deleting the list of objects in local
            object store or all object stores.
    