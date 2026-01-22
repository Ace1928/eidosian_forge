import weakref
from .. import util
from ..orm import attributes
from ..orm import base as orm_base
from ..orm import collections
from ..orm import exc as orm_exc
from ..orm import instrumentation as orm_instrumentation
from ..orm import util as orm_util
from ..orm.instrumentation import _default_dict_getter
from ..orm.instrumentation import _default_manager_getter
from ..orm.instrumentation import _default_opt_manager_getter
from ..orm.instrumentation import _default_state_getter
from ..orm.instrumentation import ClassManager
from ..orm.instrumentation import InstrumentationFactory
class InstrumentationManager:
    """User-defined class instrumentation extension.

    :class:`.InstrumentationManager` can be subclassed in order
    to change
    how class instrumentation proceeds. This class exists for
    the purposes of integration with other object management
    frameworks which would like to entirely modify the
    instrumentation methodology of the ORM, and is not intended
    for regular usage.  For interception of class instrumentation
    events, see :class:`.InstrumentationEvents`.

    The API for this class should be considered as semi-stable,
    and may change slightly with new releases.

    """

    def __init__(self, class_):
        pass

    def manage(self, class_, manager):
        setattr(class_, '_default_class_manager', manager)

    def unregister(self, class_, manager):
        delattr(class_, '_default_class_manager')

    def manager_getter(self, class_):

        def get(cls):
            return cls._default_class_manager
        return get

    def instrument_attribute(self, class_, key, inst):
        pass

    def post_configure_attribute(self, class_, key, inst):
        pass

    def install_descriptor(self, class_, key, inst):
        setattr(class_, key, inst)

    def uninstall_descriptor(self, class_, key):
        delattr(class_, key)

    def install_member(self, class_, key, implementation):
        setattr(class_, key, implementation)

    def uninstall_member(self, class_, key):
        delattr(class_, key)

    def instrument_collection_class(self, class_, key, collection_class):
        return collections.prepare_instrumentation(collection_class)

    def get_instance_dict(self, class_, instance):
        return instance.__dict__

    def initialize_instance_dict(self, class_, instance):
        pass

    def install_state(self, class_, instance, state):
        setattr(instance, '_default_state', state)

    def remove_state(self, class_, instance):
        delattr(instance, '_default_state')

    def state_getter(self, class_):
        return lambda instance: getattr(instance, '_default_state')

    def dict_getter(self, class_):
        return lambda inst: self.get_instance_dict(class_, inst)