from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def unbound_method(method):
    """
    Allow to use single-argument functions as input or output processors
    (no need to define an unused first 'self' argument)
    """
    with suppress(AttributeError):
        if '.' not in method.__qualname__:
            return method.__func__
    return method