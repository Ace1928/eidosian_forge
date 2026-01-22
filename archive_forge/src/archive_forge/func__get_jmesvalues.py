from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def _get_jmesvalues(self, jmess):
    self._check_selector_method()
    jmess = arg_to_iter(jmess)
    if not hasattr(self.selector, 'jmespath'):
        raise AttributeError('Please install parsel >= 1.8.1 to get jmespath support')
    return flatten((self.selector.jmespath(jmes).getall() for jmes in jmess))