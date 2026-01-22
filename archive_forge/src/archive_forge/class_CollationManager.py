import locale
import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union
from urllib.parse import urljoin, urlsplit
from .exceptions import xpath_error
class CollationManager(context_class_base):
    """
    Context Manager for collations. Provide helper operators as methods.
    """
    lc_collate: Union[None, str, Tuple[Optional[str], Optional[str]]]
    fallback: bool = False
    _current_lc_collate: Optional[Tuple[Optional[str], Optional[str]]] = None

    def __init__(self, collation: Optional[str], token: Optional['XPathToken']=None) -> None:
        self.collation = collation
        self.token = token
        self.strcoll = locale.strcoll
        self.strxfrm = locale.strxfrm
        if collation is None:
            msg = 'collation cannot be an empty sequence'
            raise xpath_error('XPTY0004', msg, self.token)
        elif not urlsplit(collation).scheme and token is not None:
            base_uri = token.parser.base_uri
            if base_uri:
                collation = urljoin(base_uri, collation)
        if collation == UNICODE_CODEPOINT_COLLATION:
            self.lc_collate = None
            self.strcoll = unicode_codepoint_strcoll
            self.strxfrm = unicode_codepoint_strxfrm
        elif collation == HTML_ASCII_CASE_INSENSITIVE_COLLATION:
            self.lc_collate = None
            self.strcoll = case_insensitive_strcoll
            self.strxfrm = case_insensitive_strxfrm
        elif collation == XQUERY_TEST_SUITE_CASEBLIND_COLLATION:
            self.lc_collate = None
            self.strcoll = case_insensitive_strcoll
            self.strxfrm = case_insensitive_strxfrm
        elif collation.startswith(UNICODE_COLLATION_BASE_URI):
            self.lc_collate = 'en_US.UTF-8'
            self.fallback = True
            for param in urlsplit(collation).query.split(';'):
                assert isinstance(param, str)
                if param.startswith('lang='):
                    lang = param[5:]
                    self.lc_collate = lang if '.' in lang else (lang, 'UTF-8')
                elif param.startswith('fallback='):
                    if param.endswith('yes'):
                        self.fallback = True
                    elif param.endswith('no'):
                        self.fallback = False
        else:
            self.lc_collate = collation

    def __enter__(self) -> 'CollationManager':
        if self.lc_collate is not None:
            _locale_collate_lock.acquire()
            self._current_lc_collate = locale.getlocale(locale.LC_COLLATE)
            try:
                locale.setlocale(locale.LC_COLLATE, self.lc_collate)
            except locale.Error:
                if not self.fallback:
                    self._current_lc_collate = None
                    _locale_collate_lock.release()
                    msg = f'Unsupported collation {self.collation!r}'
                    raise xpath_error('FOCH0002', msg, self.token) from None
                locale.setlocale(locale.LC_COLLATE, 'en_US.UTF-8')
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        if self._current_lc_collate is not None:
            locale.setlocale(locale.LC_COLLATE, self._current_lc_collate)
            self._current_lc_collate = None
            _locale_collate_lock.release()

    def eq(self, a: Any, b: Any) -> bool:
        if not isinstance(a, str) or not isinstance(b, str):
            return bool(a == b)
        return self.strcoll(a, b) == 0

    def ne(self, a: Any, b: Any) -> bool:
        if not isinstance(a, str) or not isinstance(b, str):
            return bool(a != b)
        return self.strcoll(a, b) != 0

    def contains(self, a: str, b: str) -> bool:
        return self.strxfrm(b) in self.strxfrm(a)

    def find(self, a: str, b: str) -> int:
        return self.strxfrm(a).find(self.strxfrm(b))

    def startswith(self, a: str, b: str) -> int:
        return self.strxfrm(a).startswith(self.strxfrm(b))

    def endswith(self, a: str, b: str) -> int:
        return self.strxfrm(a).endswith(self.strxfrm(b))