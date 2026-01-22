import six
from ebooklib.plugins.base import BasePlugin
from ebooklib.utils import parse_html_string
class SyntaxPlugin(BasePlugin):
    NAME = 'Check HTML syntax'

    def html_before_write(self, book, chapter):
        from lxml import etree
        try:
            tree = parse_html_string(chapter.content)
        except:
            return
        root = tree.getroottree()
        for tag in DEPRECATED_TAGS:
            etree.strip_tags(root, tag)
        head = tree.find('head')
        if head is not None and len(head) != 0:
            for _item in head:
                if _item.tag == 'base':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['href', 'target'])
                elif _item.tag == 'link':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['href', 'crossorigin', 'rel', 'media', 'hreflang', 'type', 'sizes'])
                elif _item.tag == 'title':
                    if _item.text == '':
                        head.remove(_item)
                elif _item.tag == 'meta':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['name', 'http-equiv', 'content', 'charset'])
                    head.remove(_item)
                elif _item.tag == 'script':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['src', 'type', 'charset', 'async', 'defer', 'crossorigin'])
                elif _item.tag == 'source':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['src', 'type', 'media'])
                elif _item.tag == 'style':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['media', 'type', 'scoped'])
                else:
                    leave_only(_item, ATTRIBUTES_GLOBAL)
        if len(root.find('body')) != 0:
            body = tree.find('body')
            for _item in body.iter():
                if _item.tag == 'a':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['href', 'target', 'download', 'rel', 'hreflang', 'type'])
                elif _item.tag == 'area':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['alt', 'coords', 'shape', 'href', 'target', 'download', 'rel', 'hreflang', 'type'])
                elif _item.tag == 'audio':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['src', 'crossorigin', 'preload', 'autoplay', 'mediagroup', 'loop', 'muted', 'controls'])
                elif _item.tag == 'blockquote':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['cite'])
                elif _item.tag == 'button':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['autofocus', 'disabled', 'form', 'formaction', 'formenctype', 'formmethod', 'formnovalidate', 'formtarget', 'name', 'type', 'value', 'menu'])
                elif _item.tag == 'canvas':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['width', 'height'])
                elif _item.tag == 'canvas':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['width', 'height'])
                elif _item.tag == 'del':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['cite', 'datetime'])
                elif _item.tag == 'details':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['open'])
                elif _item.tag == 'embed':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['src', 'type', 'width', 'height'])
                elif _item.tag == 'fieldset':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['disable', 'form', 'name'])
                elif _item.tag == 'details':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['accept-charset', 'action', 'autocomplete', 'enctype', 'method', 'name', 'novalidate', 'target'])
                elif _item.tag == 'iframe':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['src', 'srcdoc', 'name', 'sandbox', 'seamless', 'allowfullscreen', 'width', 'height'])
                elif _item.tag == 'img':
                    _src = _item.get('src', '').lower()
                    if _src.startswith('http://') or _src.startswith('https://'):
                        if 'remote-resources' not in chapter.properties:
                            chapter.properties.append('remote-resources')
                            from ebooklib import epub
                            _img = epub.EpubImage(file_name=_item.get('src'))
                            book.add_item(_img)
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['alt', 'src', 'crossorigin', 'usemap', 'ismap', 'width', 'height'])
                elif _item.tag == 'input':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['accept', 'alt', 'autocomplete', 'autofocus', 'checked', 'dirname', 'disabled', 'form', 'formaction', 'formenctype', 'formmethod', 'formnovalidate', 'formtarget', 'height', 'inputmode', 'list', 'max', 'maxlength', 'min', 'multiple', 'name', 'pattern', 'placeholder', 'readonly', 'required', 'size', 'src', 'steptype', 'value', 'width'])
                elif _item.tag == 'ins':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['cite', 'datetime'])
                elif _item.tag == 'keygen':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['autofocus', 'challenge', 'disabled', 'form', 'keytype', 'name'])
                elif _item.tag == 'label':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['form', 'for'])
                elif _item.tag == 'label':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['form', 'for'])
                elif _item.tag == 'map':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['name'])
                elif _item.tag == 'menu':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['type', 'label'])
                elif _item.tag == 'object':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['data', 'type', 'typemustmatch', 'name', 'usemap', 'form', 'width', 'height'])
                elif _item.tag == 'ol':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['reversed', 'start', 'type'])
                elif _item.tag == 'optgroup':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['disabled', 'label'])
                elif _item.tag == 'option':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['disabled', 'label', 'selected', 'value'])
                elif _item.tag == 'output':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['for', 'form', 'name'])
                elif _item.tag == 'param':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['name', 'value'])
                elif _item.tag == 'progress':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['value', 'max'])
                elif _item.tag == 'q':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['cite'])
                elif _item.tag == 'select':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['autofocus', 'disabled', 'form', 'multiple', 'name', 'required', 'size'])
                elif _item.tag == 'table':
                    if _item.get('border', None):
                        if _item.get('border') == '0':
                            _item.set('border', '')
                    if _item.get('summary', None):
                        _caption = etree.Element('caption', {})
                        _caption.text = _item.get('summary')
                        _item.insert(0, _caption)
                        del _item.attrib['summary']
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['border', 'sortable'])
                elif _item.tag == 'dl':
                    _d = _item.find('dd')
                    if _d is not None and len(_d) == 0:
                        pass
                elif _item.tag == 'td':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['colspan', 'rowspan', 'headers'])
                elif _item.tag == 'textarea':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['autocomplete', 'autofocus', 'cols', 'dirname', 'disabled', 'form', 'inputmode', 'maxlength', 'name', 'placeholder', 'readonly', 'required', 'rows', 'wrap'])
                elif _item.tag in ['col', 'colgroup']:
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['span'])
                elif _item.tag == 'th':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['colspan', 'rowspan', 'headers', 'scope', 'abbr', 'sorted'])
                elif _item.tag in ['time']:
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['datetime'])
                elif _item.tag in ['track']:
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['kind', 'src', 'srclang', 'label', 'default'])
                elif _item.tag == 'video':
                    leave_only(_item, ATTRIBUTES_GLOBAL + ['src', 'crossorigin', 'poster', 'preload', 'autoplay', 'mediagroup', 'loop', 'muted', 'controls', 'width', 'height'])
                elif _item.tag == 'svg':
                    if 'svg' not in chapter.properties:
                        chapter.properties.append('svg')
                    if _item.get('viewbox', None):
                        del _item.attrib['viewbox']
                    if _item.get('preserveaspectratio', None):
                        del _item.attrib['preserveaspectratio']
                else:
                    for _attr in six.iterkeys(_item.attrib):
                        if _attr not in ATTRIBUTES_GLOBAL:
                            del _item.attrib[_attr]
        chapter.content = etree.tostring(tree, pretty_print=True, encoding='utf-8', xml_declaration=True)
        return chapter.content