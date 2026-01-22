from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
class DefaultErrorCreator:
    insert_before = True
    block_inside = True
    error_container_tag = 'div'
    error_message_class = 'error-message'
    error_block_class = 'error-block'
    default_message = 'Invalid'

    def __init__(self, **kw):
        for name, value in kw.items():
            if not hasattr(self, name):
                raise TypeError('Unexpected keyword argument: %s' % name)
            setattr(self, name, value)

    def __call__(self, el, is_block, message):
        error_el = el.makeelement(self.error_container_tag)
        if self.error_message_class:
            error_el.set('class', self.error_message_class)
        if is_block and self.error_block_class:
            error_el.set('class', error_el.get('class', '') + ' ' + self.error_block_class)
        if message is None or message == '':
            message = self.default_message
        if isinstance(message, ElementBase):
            error_el.append(message)
        else:
            assert isinstance(message, basestring), 'Bad message; should be a string or element: %r' % message
            error_el.text = message or self.default_message
        if is_block and self.block_inside:
            if self.insert_before:
                error_el.tail = el.text
                el.text = None
                el.insert(0, error_el)
            else:
                el.append(error_el)
        else:
            parent = el.getparent()
            pos = parent.index(el)
            if self.insert_before:
                parent.insert(pos, error_el)
            else:
                error_el.tail = el.tail
                el.tail = None
                parent.insert(pos + 1, error_el)