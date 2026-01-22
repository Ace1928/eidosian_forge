from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _insert_error(el, error, error_class, error_creator):
    if _nons(el.tag) in defs.empty_tags or _nons(el.tag) == 'textarea':
        is_block = False
    else:
        is_block = True
    if _nons(el.tag) != 'form' and error_class:
        _add_class(el, error_class)
    if el.get('id'):
        labels = _label_for_xpath(el, for_id=el.get('id'))
        if labels:
            for label in labels:
                _add_class(label, error_class)
    error_creator(el, is_block, error)