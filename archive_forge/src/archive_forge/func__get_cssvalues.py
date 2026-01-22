from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def _get_cssvalues(self, csss):
    self._check_selector_method()
    csss = arg_to_iter(csss)
    return flatten((self.selector.css(css).getall() for css in csss))