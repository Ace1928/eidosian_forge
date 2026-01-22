from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _fill_form(el, values):
    counts = {}
    if hasattr(values, 'mixed'):
        values = values.mixed()
    inputs = _input_xpath(el)
    for input in inputs:
        name = input.get('name')
        if not name:
            continue
        if _takes_multiple(input):
            value = values.get(name, [])
            if not isinstance(value, (list, tuple)):
                value = [value]
            _fill_multiple(input, value)
        elif name not in values:
            continue
        else:
            index = counts.get(name, 0)
            counts[name] = index + 1
            value = values[name]
            if isinstance(value, (list, tuple)):
                try:
                    value = value[index]
                except IndexError:
                    continue
            elif index > 0:
                continue
            _fill_single(input, value)