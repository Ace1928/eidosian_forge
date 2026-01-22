from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
class CSSParser:
    """Parse CSS selectors."""
    css_tokens = (SelectorPattern('pseudo_close', PAT_PSEUDO_CLOSE), SpecialPseudoPattern((('pseudo_contains', (':contains', ':-soup-contains', ':-soup-contains-own'), PAT_PSEUDO_CONTAINS, SelectorPattern), ('pseudo_nth_child', (':nth-child', ':nth-last-child'), PAT_PSEUDO_NTH_CHILD, SelectorPattern), ('pseudo_nth_type', (':nth-of-type', ':nth-last-of-type'), PAT_PSEUDO_NTH_TYPE, SelectorPattern), ('pseudo_lang', (':lang',), PAT_PSEUDO_LANG, SelectorPattern), ('pseudo_dir', (':dir',), PAT_PSEUDO_DIR, SelectorPattern))), SelectorPattern('pseudo_class_custom', PAT_PSEUDO_CLASS_CUSTOM), SelectorPattern('pseudo_class', PAT_PSEUDO_CLASS), SelectorPattern('pseudo_element', PAT_PSEUDO_ELEMENT), SelectorPattern('at_rule', PAT_AT_RULE), SelectorPattern('id', PAT_ID), SelectorPattern('class', PAT_CLASS), SelectorPattern('tag', PAT_TAG), SelectorPattern('attribute', PAT_ATTR), SelectorPattern('combine', PAT_COMBINE))

    def __init__(self, selector: str, custom: dict[str, str | ct.SelectorList] | None=None, flags: int=0) -> None:
        """Initialize."""
        self.pattern = selector.replace('\x00', '�')
        self.flags = flags
        self.debug = self.flags & util.DEBUG
        self.custom = {} if custom is None else custom

    def parse_attribute_selector(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Create attribute selector from the returned regex match."""
        inverse = False
        op = m.group('cmp')
        case = util.lower(m.group('case')) if m.group('case') else None
        ns = css_unescape(m.group('attr_ns')[:-1]) if m.group('attr_ns') else ''
        attr = css_unescape(m.group('attr_name'))
        is_type = False
        pattern2 = None
        value = ''
        if case:
            flags = (re.I if case == 'i' else 0) | re.DOTALL
        elif util.lower(attr) == 'type':
            flags = re.I | re.DOTALL
            is_type = True
        else:
            flags = re.DOTALL
        if op:
            if m.group('value').startswith(('"', "'")):
                value = css_unescape(m.group('value')[1:-1], True)
            else:
                value = css_unescape(m.group('value'))
        if not op:
            pattern = None
        elif op.startswith('^'):
            pattern = re.compile('^%s.*' % re.escape(value), flags)
        elif op.startswith('$'):
            pattern = re.compile('.*?%s$' % re.escape(value), flags)
        elif op.startswith('*'):
            pattern = re.compile('.*?%s.*' % re.escape(value), flags)
        elif op.startswith('~'):
            value = '[^\\s\\S]' if not value or RE_WS.search(value) else re.escape(value)
            pattern = re.compile('.*?(?:(?<=^)|(?<=[ \\t\\r\\n\\f]))%s(?=(?:[ \\t\\r\\n\\f]|$)).*' % value, flags)
        elif op.startswith('|'):
            pattern = re.compile('^%s(?:-.*)?$' % re.escape(value), flags)
        else:
            pattern = re.compile('^%s$' % re.escape(value), flags)
            if op.startswith('!'):
                inverse = True
        if is_type and pattern:
            pattern2 = re.compile(pattern.pattern)
        sel_attr = ct.SelectorAttribute(attr, ns, pattern, pattern2)
        if inverse:
            sub_sel = _Selector()
            sub_sel.attributes.append(sel_attr)
            not_list = ct.SelectorList([sub_sel.freeze()], True, False)
            sel.selectors.append(not_list)
        else:
            sel.attributes.append(sel_attr)
        has_selector = True
        return has_selector

    def parse_tag_pattern(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse tag pattern from regex match."""
        prefix = css_unescape(m.group('tag_ns')[:-1]) if m.group('tag_ns') else None
        tag = css_unescape(m.group('tag_name'))
        sel.tag = ct.SelectorTag(tag, prefix)
        has_selector = True
        return has_selector

    def parse_pseudo_class_custom(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """
        Parse custom pseudo class alias.

        Compile custom selectors as we need them. When compiling a custom selector,
        set it to `None` in the dictionary so we can avoid an infinite loop.
        """
        pseudo = util.lower(css_unescape(m.group('name')))
        selector = self.custom.get(pseudo)
        if selector is None:
            raise SelectorSyntaxError(f"Undefined custom selector '{pseudo}' found at position {m.end(0)}", self.pattern, m.end(0))
        if not isinstance(selector, ct.SelectorList):
            del self.custom[pseudo]
            selector = CSSParser(selector, custom=self.custom, flags=self.flags).process_selectors(flags=FLG_PSEUDO)
            self.custom[pseudo] = selector
        sel.selectors.append(selector)
        has_selector = True
        return has_selector

    def parse_pseudo_class(self, sel: _Selector, m: Match[str], has_selector: bool, iselector: Iterator[tuple[str, Match[str]]], is_html: bool) -> tuple[bool, bool]:
        """Parse pseudo class."""
        complex_pseudo = False
        pseudo = util.lower(css_unescape(m.group('name')))
        if m.group('open'):
            complex_pseudo = True
        if complex_pseudo and pseudo in PSEUDO_COMPLEX:
            has_selector = self.parse_pseudo_open(sel, pseudo, has_selector, iselector, m.end(0))
        elif not complex_pseudo and pseudo in PSEUDO_SIMPLE:
            if pseudo == ':root':
                sel.flags |= ct.SEL_ROOT
            elif pseudo == ':defined':
                sel.flags |= ct.SEL_DEFINED
                is_html = True
            elif pseudo == ':scope':
                sel.flags |= ct.SEL_SCOPE
            elif pseudo == ':empty':
                sel.flags |= ct.SEL_EMPTY
            elif pseudo in (':link', ':any-link'):
                sel.selectors.append(CSS_LINK)
            elif pseudo == ':checked':
                sel.selectors.append(CSS_CHECKED)
            elif pseudo == ':default':
                sel.selectors.append(CSS_DEFAULT)
            elif pseudo == ':indeterminate':
                sel.selectors.append(CSS_INDETERMINATE)
            elif pseudo == ':disabled':
                sel.selectors.append(CSS_DISABLED)
            elif pseudo == ':enabled':
                sel.selectors.append(CSS_ENABLED)
            elif pseudo == ':required':
                sel.selectors.append(CSS_REQUIRED)
            elif pseudo == ':optional':
                sel.selectors.append(CSS_OPTIONAL)
            elif pseudo == ':read-only':
                sel.selectors.append(CSS_READ_ONLY)
            elif pseudo == ':read-write':
                sel.selectors.append(CSS_READ_WRITE)
            elif pseudo == ':in-range':
                sel.selectors.append(CSS_IN_RANGE)
            elif pseudo == ':out-of-range':
                sel.selectors.append(CSS_OUT_OF_RANGE)
            elif pseudo == ':placeholder-shown':
                sel.selectors.append(CSS_PLACEHOLDER_SHOWN)
            elif pseudo == ':first-child':
                sel.nth.append(ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()))
            elif pseudo == ':last-child':
                sel.nth.append(ct.SelectorNth(1, False, 0, False, True, ct.SelectorList()))
            elif pseudo == ':first-of-type':
                sel.nth.append(ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()))
            elif pseudo == ':last-of-type':
                sel.nth.append(ct.SelectorNth(1, False, 0, True, True, ct.SelectorList()))
            elif pseudo == ':only-child':
                sel.nth.extend([ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()), ct.SelectorNth(1, False, 0, False, True, ct.SelectorList())])
            elif pseudo == ':only-of-type':
                sel.nth.extend([ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()), ct.SelectorNth(1, False, 0, True, True, ct.SelectorList())])
            has_selector = True
        elif complex_pseudo and pseudo in PSEUDO_COMPLEX_NO_MATCH:
            self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
            sel.no_match = True
            has_selector = True
        elif not complex_pseudo and pseudo in PSEUDO_SIMPLE_NO_MATCH:
            sel.no_match = True
            has_selector = True
        elif pseudo in PSEUDO_SUPPORTED:
            raise SelectorSyntaxError(f"Invalid syntax for pseudo class '{pseudo}'", self.pattern, m.start(0))
        else:
            raise NotImplementedError(f"'{pseudo}' pseudo-class is not implemented at this time")
        return (has_selector, is_html)

    def parse_pseudo_nth(self, sel: _Selector, m: Match[str], has_selector: bool, iselector: Iterator[tuple[str, Match[str]]]) -> bool:
        """Parse `nth` pseudo."""
        mdict = m.groupdict()
        if mdict.get('pseudo_nth_child'):
            postfix = '_child'
        else:
            postfix = '_type'
        mdict['name'] = util.lower(css_unescape(mdict['name']))
        content = util.lower(mdict.get('nth' + postfix))
        if content == 'even':
            s1 = 2
            s2 = 0
            var = True
        elif content == 'odd':
            s1 = 2
            s2 = 1
            var = True
        else:
            nth_parts = cast(Match[str], RE_NTH.match(content))
            _s1 = '-' if nth_parts.group('s1') and nth_parts.group('s1') == '-' else ''
            a = nth_parts.group('a')
            var = a.endswith('n')
            if a.startswith('n'):
                _s1 += '1'
            elif var:
                _s1 += a[:-1]
            else:
                _s1 += a
            _s2 = '-' if nth_parts.group('s2') and nth_parts.group('s2') == '-' else ''
            if nth_parts.group('b'):
                _s2 += nth_parts.group('b')
            else:
                _s2 = '0'
            s1 = int(_s1, 10)
            s2 = int(_s2, 10)
        pseudo_sel = mdict['name']
        if postfix == '_child':
            if m.group('of'):
                nth_sel = self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
            else:
                nth_sel = CSS_NTH_OF_S_DEFAULT
            if pseudo_sel == ':nth-child':
                sel.nth.append(ct.SelectorNth(s1, var, s2, False, False, nth_sel))
            elif pseudo_sel == ':nth-last-child':
                sel.nth.append(ct.SelectorNth(s1, var, s2, False, True, nth_sel))
        elif pseudo_sel == ':nth-of-type':
            sel.nth.append(ct.SelectorNth(s1, var, s2, True, False, ct.SelectorList()))
        elif pseudo_sel == ':nth-last-of-type':
            sel.nth.append(ct.SelectorNth(s1, var, s2, True, True, ct.SelectorList()))
        has_selector = True
        return has_selector

    def parse_pseudo_open(self, sel: _Selector, name: str, has_selector: bool, iselector: Iterator[tuple[str, Match[str]]], index: int) -> bool:
        """Parse pseudo with opening bracket."""
        flags = FLG_PSEUDO | FLG_OPEN
        if name == ':not':
            flags |= FLG_NOT
        elif name == ':has':
            flags |= FLG_RELATIVE
        elif name in (':where', ':is'):
            flags |= FLG_FORGIVE
        sel.selectors.append(self.parse_selectors(iselector, index, flags))
        has_selector = True
        return has_selector

    def parse_has_combinator(self, sel: _Selector, m: Match[str], has_selector: bool, selectors: list[_Selector], rel_type: str, index: int) -> tuple[bool, _Selector, str]:
        """Parse combinator tokens."""
        combinator = m.group('relation').strip()
        if not combinator:
            combinator = WS_COMBINATOR
        if combinator == COMMA_COMBINATOR:
            sel.rel_type = rel_type
            selectors[-1].relations.append(sel)
            rel_type = ':' + WS_COMBINATOR
            selectors.append(_Selector())
        else:
            if has_selector:
                sel.rel_type = rel_type
                selectors[-1].relations.append(sel)
            elif rel_type[1:] != WS_COMBINATOR:
                raise SelectorSyntaxError(f'The multiple combinators at position {index}', self.pattern, index)
            rel_type = ':' + combinator
        sel = _Selector()
        has_selector = False
        return (has_selector, sel, rel_type)

    def parse_combinator(self, sel: _Selector, m: Match[str], has_selector: bool, selectors: list[_Selector], relations: list[_Selector], is_pseudo: bool, is_forgive: bool, index: int) -> tuple[bool, _Selector]:
        """Parse combinator tokens."""
        combinator = m.group('relation').strip()
        if not combinator:
            combinator = WS_COMBINATOR
        if not has_selector:
            if not is_forgive or combinator != COMMA_COMBINATOR:
                raise SelectorSyntaxError(f"The combinator '{combinator}' at position {index}, must have a selector before it", self.pattern, index)
            if combinator == COMMA_COMBINATOR:
                sel.no_match = True
                del relations[:]
                selectors.append(sel)
        elif combinator == COMMA_COMBINATOR:
            if not sel.tag and (not is_pseudo):
                sel.tag = ct.SelectorTag('*', None)
            sel.relations.extend(relations)
            selectors.append(sel)
            del relations[:]
        else:
            sel.relations.extend(relations)
            sel.rel_type = combinator
            del relations[:]
            relations.append(sel)
        sel = _Selector()
        has_selector = False
        return (has_selector, sel)

    def parse_class_id(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse HTML classes and ids."""
        selector = m.group(0)
        if selector.startswith('.'):
            sel.classes.append(css_unescape(selector[1:]))
        else:
            sel.ids.append(css_unescape(selector[1:]))
        has_selector = True
        return has_selector

    def parse_pseudo_contains(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse contains."""
        pseudo = util.lower(css_unescape(m.group('name')))
        if pseudo == ':contains':
            warnings.warn("The pseudo class ':contains' is deprecated, ':-soup-contains' should be used moving forward.", FutureWarning)
        contains_own = pseudo == ':-soup-contains-own'
        values = css_unescape(m.group('values'))
        patterns = []
        for token in RE_VALUES.finditer(values):
            if token.group('split'):
                continue
            value = token.group('value')
            if value.startswith(("'", '"')):
                value = css_unescape(value[1:-1], True)
            else:
                value = css_unescape(value)
            patterns.append(value)
        sel.contains.append(ct.SelectorContains(patterns, contains_own))
        has_selector = True
        return has_selector

    def parse_pseudo_lang(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse pseudo language."""
        values = m.group('values')
        patterns = []
        for token in RE_VALUES.finditer(values):
            if token.group('split'):
                continue
            value = token.group('value')
            if value.startswith(('"', "'")):
                value = css_unescape(value[1:-1], True)
            else:
                value = css_unescape(value)
            patterns.append(value)
        sel.lang.append(ct.SelectorLang(patterns))
        has_selector = True
        return has_selector

    def parse_pseudo_dir(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse pseudo direction."""
        value = ct.SEL_DIR_LTR if util.lower(m.group('dir')) == 'ltr' else ct.SEL_DIR_RTL
        sel.flags |= value
        has_selector = True
        return has_selector

    def parse_selectors(self, iselector: Iterator[tuple[str, Match[str]]], index: int=0, flags: int=0) -> ct.SelectorList:
        """Parse selectors."""
        sel = _Selector()
        selectors = []
        has_selector = False
        closed = False
        relations = []
        rel_type = ':' + WS_COMBINATOR
        is_open = bool(flags & FLG_OPEN)
        is_pseudo = bool(flags & FLG_PSEUDO)
        is_relative = bool(flags & FLG_RELATIVE)
        is_not = bool(flags & FLG_NOT)
        is_html = bool(flags & FLG_HTML)
        is_default = bool(flags & FLG_DEFAULT)
        is_indeterminate = bool(flags & FLG_INDETERMINATE)
        is_in_range = bool(flags & FLG_IN_RANGE)
        is_out_of_range = bool(flags & FLG_OUT_OF_RANGE)
        is_placeholder_shown = bool(flags & FLG_PLACEHOLDER_SHOWN)
        is_forgive = bool(flags & FLG_FORGIVE)
        if self.debug:
            if is_pseudo:
                print('    is_pseudo: True')
            if is_open:
                print('    is_open: True')
            if is_relative:
                print('    is_relative: True')
            if is_not:
                print('    is_not: True')
            if is_html:
                print('    is_html: True')
            if is_default:
                print('    is_default: True')
            if is_indeterminate:
                print('    is_indeterminate: True')
            if is_in_range:
                print('    is_in_range: True')
            if is_out_of_range:
                print('    is_out_of_range: True')
            if is_placeholder_shown:
                print('    is_placeholder_shown: True')
            if is_forgive:
                print('    is_forgive: True')
        if is_relative:
            selectors.append(_Selector())
        try:
            while True:
                key, m = next(iselector)
                if key == 'at_rule':
                    raise NotImplementedError(f'At-rules found at position {m.start(0)}')
                elif key == 'pseudo_class_custom':
                    has_selector = self.parse_pseudo_class_custom(sel, m, has_selector)
                elif key == 'pseudo_class':
                    has_selector, is_html = self.parse_pseudo_class(sel, m, has_selector, iselector, is_html)
                elif key == 'pseudo_element':
                    raise NotImplementedError(f'Pseudo-element found at position {m.start(0)}')
                elif key == 'pseudo_contains':
                    has_selector = self.parse_pseudo_contains(sel, m, has_selector)
                elif key in ('pseudo_nth_type', 'pseudo_nth_child'):
                    has_selector = self.parse_pseudo_nth(sel, m, has_selector, iselector)
                elif key == 'pseudo_lang':
                    has_selector = self.parse_pseudo_lang(sel, m, has_selector)
                elif key == 'pseudo_dir':
                    has_selector = self.parse_pseudo_dir(sel, m, has_selector)
                    is_html = True
                elif key == 'pseudo_close':
                    if not has_selector:
                        if not is_forgive:
                            raise SelectorSyntaxError(f'Expected a selector at position {m.start(0)}', self.pattern, m.start(0))
                        sel.no_match = True
                    if is_open:
                        closed = True
                        break
                    else:
                        raise SelectorSyntaxError(f'Unmatched pseudo-class close at position {m.start(0)}', self.pattern, m.start(0))
                elif key == 'combine':
                    if is_relative:
                        has_selector, sel, rel_type = self.parse_has_combinator(sel, m, has_selector, selectors, rel_type, index)
                    else:
                        has_selector, sel = self.parse_combinator(sel, m, has_selector, selectors, relations, is_pseudo, is_forgive, index)
                elif key == 'attribute':
                    has_selector = self.parse_attribute_selector(sel, m, has_selector)
                elif key == 'tag':
                    if has_selector:
                        raise SelectorSyntaxError(f'Tag name found at position {m.start(0)} instead of at the start', self.pattern, m.start(0))
                    has_selector = self.parse_tag_pattern(sel, m, has_selector)
                elif key in ('class', 'id'):
                    has_selector = self.parse_class_id(sel, m, has_selector)
                index = m.end(0)
        except StopIteration:
            pass
        if is_open and (not closed):
            raise SelectorSyntaxError(f'Unclosed pseudo-class at position {index}', self.pattern, index)
        if has_selector:
            if not sel.tag and (not is_pseudo):
                sel.tag = ct.SelectorTag('*', None)
            if is_relative:
                sel.rel_type = rel_type
                selectors[-1].relations.append(sel)
            else:
                sel.relations.extend(relations)
                del relations[:]
                selectors.append(sel)
        elif is_forgive and (not selectors or not relations):
            sel.no_match = True
            del relations[:]
            selectors.append(sel)
            has_selector = True
        if not has_selector:
            raise SelectorSyntaxError(f'Expected a selector at position {index}', self.pattern, index)
        if is_default:
            selectors[-1].flags = ct.SEL_DEFAULT
        if is_indeterminate:
            selectors[-1].flags = ct.SEL_INDETERMINATE
        if is_in_range:
            selectors[-1].flags = ct.SEL_IN_RANGE
        if is_out_of_range:
            selectors[-1].flags = ct.SEL_OUT_OF_RANGE
        if is_placeholder_shown:
            selectors[-1].flags = ct.SEL_PLACEHOLDER_SHOWN
        return ct.SelectorList([s.freeze() for s in selectors], is_not, is_html)

    def selector_iter(self, pattern: str) -> Iterator[tuple[str, Match[str]]]:
        """Iterate selector tokens."""
        m = RE_WS_BEGIN.search(pattern)
        index = m.end(0) if m else 0
        m = RE_WS_END.search(pattern)
        end = m.start(0) - 1 if m else len(pattern) - 1
        if self.debug:
            print(f'## PARSING: {pattern!r}')
        while index <= end:
            m = None
            for v in self.css_tokens:
                m = v.match(pattern, index, self.flags)
                if m:
                    name = v.get_name()
                    if self.debug:
                        print(f"TOKEN: '{name}' --> {m.group(0)!r} at position {m.start(0)}")
                    index = m.end(0)
                    yield (name, m)
                    break
            if m is None:
                c = pattern[index]
                if c == '[':
                    msg = f'Malformed attribute selector at position {index}'
                elif c == '.':
                    msg = f'Malformed class selector at position {index}'
                elif c == '#':
                    msg = f'Malformed id selector at position {index}'
                elif c == ':':
                    msg = f'Malformed pseudo-class selector at position {index}'
                else:
                    msg = f'Invalid character {c!r} position {index}'
                raise SelectorSyntaxError(msg, self.pattern, index)
        if self.debug:
            print('## END PARSING')

    def process_selectors(self, index: int=0, flags: int=0) -> ct.SelectorList:
        """Process selectors."""
        return self.parse_selectors(self.selector_iter(self.pattern), index, flags)