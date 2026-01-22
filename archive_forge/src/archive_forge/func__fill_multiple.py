from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _fill_multiple(input, value):
    type = input.get('type', '').lower()
    if type == 'checkbox':
        v = input.get('value')
        if v is None:
            if not value:
                result = False
            else:
                result = value[0]
                if isinstance(value, basestring):
                    result = result == 'on'
            _check(input, result)
        else:
            _check(input, v in value)
    elif type == 'radio':
        v = input.get('value')
        _check(input, v in value)
    else:
        assert _nons(input.tag) == 'select'
        for option in _options_xpath(input):
            v = option.get('value')
            if v is None:
                v = option.text_content()
            _select(option, v in value)