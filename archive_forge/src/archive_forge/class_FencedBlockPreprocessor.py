from __future__ import annotations
from textwrap import dedent
from . import Extension
from ..preprocessors import Preprocessor
from .codehilite import CodeHilite, CodeHiliteExtension, parse_hl_lines
from .attr_list import get_attrs_and_remainder, AttrListExtension
from ..util import parseBoolValue
from ..serializers import _escape_attrib_html
import re
from typing import TYPE_CHECKING, Any, Iterable
class FencedBlockPreprocessor(Preprocessor):
    """ Find and extract fenced code blocks. """
    FENCED_BLOCK_RE = re.compile(dedent('\n            (?P<fence>^(?:~{3,}|`{3,}))[ ]*                          # opening fence\n            ((\\{(?P<attrs>[^\\n]*)\\})|                                # (optional {attrs} or\n            (\\.?(?P<lang>[\\w#.+-]*)[ ]*)?                            # optional (.)lang\n            (hl_lines=(?P<quot>"|\')(?P<hl_lines>.*?)(?P=quot)[ ]*)?) # optional hl_lines)\n            \\n                                                       # newline (end of opening fence)\n            (?P<code>.*?)(?<=\\n)                                     # the code block\n            (?P=fence)[ ]*$                                          # closing fence\n        '), re.MULTILINE | re.DOTALL | re.VERBOSE)

    def __init__(self, md: Markdown, config: dict[str, Any]):
        super().__init__(md)
        self.config = config
        self.checked_for_deps = False
        self.codehilite_conf: dict[str, Any] = {}
        self.use_attr_list = False
        self.bool_options = ['linenums', 'guess_lang', 'noclasses', 'use_pygments']

    def run(self, lines: list[str]) -> list[str]:
        """ Match and store Fenced Code Blocks in the `HtmlStash`. """
        if not self.checked_for_deps:
            for ext in self.md.registeredExtensions:
                if isinstance(ext, CodeHiliteExtension):
                    self.codehilite_conf = ext.getConfigs()
                if isinstance(ext, AttrListExtension):
                    self.use_attr_list = True
            self.checked_for_deps = True
        text = '\n'.join(lines)
        index = 0
        while 1:
            m = self.FENCED_BLOCK_RE.search(text, index)
            if m:
                lang, id, classes, config = (None, '', [], {})
                if m.group('attrs'):
                    attrs, remainder = get_attrs_and_remainder(m.group('attrs'))
                    if remainder:
                        index = m.end('attrs')
                        continue
                    id, classes, config = self.handle_attrs(attrs)
                    if len(classes):
                        lang = classes.pop(0)
                else:
                    if m.group('lang'):
                        lang = m.group('lang')
                    if m.group('hl_lines'):
                        config['hl_lines'] = parse_hl_lines(m.group('hl_lines'))
                if self.codehilite_conf and self.codehilite_conf['use_pygments'] and config.get('use_pygments', True):
                    local_config = self.codehilite_conf.copy()
                    local_config.update(config)
                    if classes:
                        local_config['css_class'] = '{} {}'.format(' '.join(classes), local_config['css_class'])
                    highliter = CodeHilite(m.group('code'), lang=lang, style=local_config.pop('pygments_style', 'default'), **local_config)
                    code = highliter.hilite(shebang=False)
                else:
                    id_attr = lang_attr = class_attr = kv_pairs = ''
                    if lang:
                        prefix = self.config.get('lang_prefix', 'language-')
                        lang_attr = f' class="{prefix}{_escape_attrib_html(lang)}"'
                    if classes:
                        class_attr = f' class="{_escape_attrib_html(' '.join(classes))}"'
                    if id:
                        id_attr = f' id="{_escape_attrib_html(id)}"'
                    if self.use_attr_list and config and (not config.get('use_pygments', False)):
                        kv_pairs = ''.join((f' {k}="{_escape_attrib_html(v)}"' for k, v in config.items() if k != 'use_pygments'))
                    code = self._escape(m.group('code'))
                    code = f'<pre{id_attr}{class_attr}><code{lang_attr}{kv_pairs}>{code}</code></pre>'
                placeholder = self.md.htmlStash.store(code)
                text = f'{text[:m.start()]}\n{placeholder}\n{text[m.end():]}'
                index = m.start() + 1 + len(placeholder)
            else:
                break
        return text.split('\n')

    def handle_attrs(self, attrs: Iterable[tuple[str, str]]) -> tuple[str, list[str], dict[str, Any]]:
        """ Return tuple: `(id, [list, of, classes], {configs})` """
        id = ''
        classes = []
        configs = {}
        for k, v in attrs:
            if k == 'id':
                id = v
            elif k == '.':
                classes.append(v)
            elif k == 'hl_lines':
                configs[k] = parse_hl_lines(v)
            elif k in self.bool_options:
                configs[k] = parseBoolValue(v, fail_on_errors=False, preserve_none=True)
            else:
                configs[k] = v
        return (id, classes, configs)

    def _escape(self, txt: str) -> str:
        """ basic html escaping """
        txt = txt.replace('&', '&amp;')
        txt = txt.replace('<', '&lt;')
        txt = txt.replace('>', '&gt;')
        txt = txt.replace('"', '&quot;')
        return txt