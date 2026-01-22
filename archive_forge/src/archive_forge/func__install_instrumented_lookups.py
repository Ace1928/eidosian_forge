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
def _install_instrumented_lookups():
    """Replace global class/object management functions
    with ExtendedInstrumentationRegistry implementations, which
    allow multiple types of class managers to be present,
    at the cost of performance.

    This function is called only by ExtendedInstrumentationRegistry
    and unit tests specific to this behavior.

    The _reinstall_default_lookups() function can be called
    after this one to re-establish the default functions.

    """
    _install_lookups(dict(instance_state=_instrumentation_factory.state_of, instance_dict=_instrumentation_factory.dict_of, manager_of_class=_instrumentation_factory.manager_of_class, opt_manager_of_class=_instrumentation_factory.opt_manager_of_class))