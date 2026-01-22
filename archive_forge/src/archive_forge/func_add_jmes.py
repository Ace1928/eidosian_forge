from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def add_jmes(self, field_name, jmes, *processors, re=None, **kw):
    """
        Similar to :meth:`ItemLoader.add_value` but receives a JMESPath selector
        instead of a value, which is used to extract a list of unicode strings
        from the selector associated with this :class:`ItemLoader`.

        See :meth:`get_jmes` for ``kwargs``.

        :param jmes: the JMESPath selector to extract data from
        :type jmes: str

        Examples::

            # HTML snippet: {"name": "Color TV"}
            loader.add_jmes('name')
            # HTML snippet: {"price": the price is $1200"}
            loader.add_jmes('price', TakeFirst(), re='the price is (.*)')
        """
    values = self._get_jmesvalues(jmes)
    self.add_value(field_name, values, *processors, re=re, **kw)