from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
class SimpleParser(Generic[T], ABC):

    def __init__(self, data: str):
        self.data = data
        self.index = 0
        self._expected = defaultdict(list)

    def __init_subclass__(cls, **kwargs):
        for n, v in cls.__dict__.items():
            if isinstance(v, FunctionType) and (not n.startswith('_')):
                setattr(cls, n, _wrap_reset(v))

    def parse(self) -> T:
        try:
            result = self.start()
        except nomatch:
            raise NoMatch(self.data, max(self._expected), self._expected[max(self._expected)]) from None
        if self.index < len(self.data):
            raise NoMatch(self.data, max(self._expected), self._expected[max(self._expected)])
        return result

    @abstractmethod
    def start(self) -> T:
        raise NotImplementedError

    def peek_static(self, expected: str) -> bool:
        l = len(expected)
        if self.data[self.index:self.index + l] == expected:
            return True
        else:
            self._expected[self.index].append(expected)
            return False

    def static(self, expected: str):
        length = len(expected)
        if self.data[self.index:self.index + length] == expected:
            self.index += length
        else:
            self._expected[self.index].append(expected)
            raise nomatch

    def static_b(self, expected: str) -> bool:
        l = len(expected)
        if self.data[self.index:self.index + l] == expected:
            self.index += l
            return True
        else:
            self._expected[self.index].append(expected)
            return False

    def anyof(self, *strings: str) -> str:
        for s in strings:
            if self.static_b(s):
                return s
        else:
            raise nomatch

    def anyof_b(self, *strings: str) -> bool:
        for s in strings:
            if self.static_b(s):
                return True
        else:
            return False

    def any(self, length: int=1) -> str:
        if self.index + length <= len(self.data):
            res = self.data[self.index:self.index + length]
            self.index += length
            return res
        else:
            self._expected[self.index].append(f'<Any {length}>')
            raise nomatch

    def any_but(self, *strings, length: int=1) -> str:
        if self.index + length <= len(self.data):
            res = self.data[self.index:self.index + length]
            if res not in strings:
                self.index += length
                return res
            else:
                self._expected[self.index].append(f'<Any {length} except {strings}>')
                raise nomatch
        else:
            self._expected[self.index].append(f'<Any {length} except {strings}>')
            raise nomatch

    def multiple(self, chars: str, mi: int, ma: Optional[int]) -> str:
        result = []
        try:
            for off in range(mi):
                if self.data[self.index + off] in chars:
                    result.append(self.data[self.index + off])
                else:
                    self._expected[self.index + off].extend(chars)
                    raise nomatch
        except IndexError:
            raise nomatch
        self.index += mi
        if ma is None:
            try:
                while True:
                    if self.data[self.index] in chars:
                        result.append(self.data[self.index])
                        self.index += 1
                    else:
                        self._expected[self.index].extend(chars)
                        break
            except IndexError:
                pass
        else:
            try:
                for _ in range(ma - mi):
                    if self.data[self.index] in chars:
                        result.append(self.data[self.index])
                        self.index += 1
                    else:
                        self._expected[self.index].extend(chars)
                        break
            except IndexError:
                pass
        return ''.join(result)