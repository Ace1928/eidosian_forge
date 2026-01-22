from __future__ import unicode_literals
from six import with_metaclass
from collections import defaultdict
import weakref
class _FilterTypeMeta(type):

    def __instancecheck__(cls, instance):
        cache = _instance_check_cache[tuple(cls.arguments_list)]

        def get():
            """ The actual test. """
            if not hasattr(instance, 'test_args'):
                return False
            return instance.test_args(*cls.arguments_list)
        try:
            return cache[instance]
        except KeyError:
            result = get()
            cache[instance] = result
            return result