import threading
import fixtures
from oslo_utils import timeutils
from oslo_utils import uuidutils
class _UUIDSentinels(object):
    """Registry of dynamically created, named, random UUID strings in regular
    (with hyphens) and similar to some keystone IDs (without hyphens) formats.

    An instance of this class will dynamically generate attributes as they are
    referenced, associating a random UUID string with each. Thereafter,
    referring to the same attribute will give the same UUID for the life of the
    instance. Plan accordingly.

    Usage::

        from oslo_utils.fixture import uuidsentinel as uuids
        from oslo_utils.fixture import keystoneidsentinel as keystids
        ...
        foo = uuids.foo
        do_a_thing(foo)
        # Referencing the same sentinel again gives the same value
        assert foo == uuids.foo
        # But a different one will be different
        assert foo != uuids.bar
        # Same approach is valid for keystoneidsentinel:
        data = create_some_data_structure(keystids.bar, var1, var2, var3)
        assert extract_bar(data) == keystids.bar
    """

    def __init__(self, is_dashed=True):
        self._sentinels = {}
        self._lock = threading.Lock()
        self.is_dashed = is_dashed

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError('Sentinels must not start with _')
        with self._lock:
            if name not in self._sentinels:
                self._sentinels[name] = uuidutils.generate_uuid(dashed=self.is_dashed)
        return self._sentinels[name]