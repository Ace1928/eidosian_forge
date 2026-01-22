import ray
import ray.cloudpickle as pickle
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class StandaloneSerializationContext:

    def _register_cloudpickle_reducer(self, cls, reducer):
        pickle.CloudPickler.dispatch[cls] = reducer

    def _unregister_cloudpickle_reducer(self, cls):
        pickle.CloudPickler.dispatch.pop(cls, None)

    def _register_cloudpickle_serializer(self, cls, custom_serializer, custom_deserializer):

        def _CloudPicklerReducer(obj):
            return (custom_deserializer, (custom_serializer(obj),))
        pickle.CloudPickler.dispatch[cls] = _CloudPicklerReducer