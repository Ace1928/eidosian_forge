import operator
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.trackable import data_structures
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.saved_model.load.VersionedTypeRegistration', v1=[])
class VersionedTypeRegistration(object):
    """Holds information about one version of a revived type."""

    def __init__(self, object_factory, version, min_producer_version, min_consumer_version, bad_consumers=None, setter=setattr):
        """Identify a revived type version.

    Args:
      object_factory: A callable which takes a SavedUserObject proto and returns
        a trackable object. Dependencies are added later via `setter`.
      version: An integer, the producer version of this wrapper type. When
        making incompatible changes to a wrapper, add a new
        `VersionedTypeRegistration` with an incremented `version`. The most
        recent version will be saved, and all registrations with a matching
        identifier will be searched for the highest compatible version to use
        when loading.
      min_producer_version: The minimum producer version number required to use
        this `VersionedTypeRegistration` when loading a proto.
      min_consumer_version: `VersionedTypeRegistration`s with a version number
        less than `min_consumer_version` will not be used to load a proto saved
        with this object. `min_consumer_version` should be set to the lowest
        version number which can successfully load protos saved by this
        object. If no matching registration is available on load, the object
        will be revived with a generic trackable type.

        `min_consumer_version` and `bad_consumers` are a blunt tool, and using
        them will generally break forward compatibility: previous versions of
        TensorFlow will revive newly saved objects as opaque trackable
        objects rather than wrapped objects. When updating wrappers, prefer
        saving new information but preserving compatibility with previous
        wrapper versions. They are, however, useful for ensuring that
        previously-released buggy wrapper versions degrade gracefully rather
        than throwing exceptions when presented with newly-saved SavedModels.
      bad_consumers: A list of consumer versions which are incompatible (in
        addition to any version less than `min_consumer_version`).
      setter: A callable with the same signature as `setattr` to use when adding
        dependencies to generated objects.
    """
        self.setter = setter
        self.identifier = None
        self._object_factory = object_factory
        self.version = version
        self._min_consumer_version = min_consumer_version
        self._min_producer_version = min_producer_version
        if bad_consumers is None:
            bad_consumers = []
        self._bad_consumers = bad_consumers

    def to_proto(self):
        """Create a SavedUserObject proto."""
        return saved_object_graph_pb2.SavedUserObject(identifier=self.identifier, version=versions_pb2.VersionDef(producer=self.version, min_consumer=self._min_consumer_version, bad_consumers=self._bad_consumers))

    def from_proto(self, proto):
        """Recreate a trackable object from a SavedUserObject proto."""
        return self._object_factory(proto)

    def should_load(self, proto):
        """Checks if this object should load the SavedUserObject `proto`."""
        if proto.identifier != self.identifier:
            return False
        if self.version < proto.version.min_consumer:
            return False
        if proto.version.producer < self._min_producer_version:
            return False
        for bad_version in proto.version.bad_consumers:
            if self.version == bad_version:
                return False
        return True