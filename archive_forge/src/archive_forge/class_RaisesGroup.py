from __future__ import annotations
import re
import sys
from typing import (
from trio._util import final
@final
class RaisesGroup(ContextManager[ExceptionInfo[BaseExceptionGroup[E]]], SuperClass[E]):
    """Contextmanager for checking for an expected `ExceptionGroup`.
    This works similar to ``pytest.raises``, and a version of it will hopefully be added upstream, after which this can be deprecated and removed. See https://github.com/pytest-dev/pytest/issues/11538


    This differs from :ref:`except* <except_star>` in that all specified exceptions must be present, *and no others*. It will similarly not catch exceptions *not* wrapped in an exceptiongroup.
    If you don't care for the nesting level of the exceptions you can pass ``strict=False``.
    It currently does not care about the order of the exceptions, so ``RaisesGroups(ValueError, TypeError)`` is equivalent to ``RaisesGroups(TypeError, ValueError)``.

    This class is not as polished as ``pytest.raises``, and is currently not as helpful in e.g. printing diffs when strings don't match, suggesting you use ``re.escape``, etc.

    Examples::

        with RaisesGroups(ValueError):
            raise ExceptionGroup("", (ValueError(),))
        with RaisesGroups(ValueError, ValueError, Matcher(TypeError, match="expected int")):
            ...
        with RaisesGroups(KeyboardInterrupt, match="hello", check=lambda x: type(x) is BaseExceptionGroup):
            ...
        with RaisesGroups(RaisesGroups(ValueError)):
            raise ExceptionGroup("", (ExceptionGroup("", (ValueError(),)),))

        with RaisesGroups(ValueError, strict=False):
            raise ExceptionGroup("", (ExceptionGroup("", (ValueError(),)),))


    `RaisesGroup.matches` can also be used directly to check a standalone exception group.


    This class is also not perfectly smart, e.g. this will likely fail currently::

        with RaisesGroups(ValueError, Matcher(ValueError, match="hello")):
            raise ExceptionGroup("", (ValueError("hello"), ValueError("goodbye")))

    even though it generally does not care about the order of the exceptions in the group.
    To avoid the above you should specify the first ValueError with a Matcher as well.

    It is also not typechecked perfectly, and that's likely not possible with the current approach. Most common usage should work without issue though.
    """
    if TYPE_CHECKING:

        def __new__(cls, *args: object, **kwargs: object) -> RaisesGroup[E]:
            ...

    def __init__(self, exception: type[E] | Matcher[E] | E, *other_exceptions: type[E] | Matcher[E] | E, strict: bool=True, match: str | Pattern[str] | None=None, check: Callable[[BaseExceptionGroup[E]], bool] | None=None):
        self.expected_exceptions: tuple[type[E] | Matcher[E] | E, ...] = (exception, *other_exceptions)
        self.strict = strict
        self.match_expr = match
        self.check = check
        self.is_baseexceptiongroup = False
        for exc in self.expected_exceptions:
            if isinstance(exc, RaisesGroup):
                if not strict:
                    raise ValueError('You cannot specify a nested structure inside a RaisesGroup with strict=False')
                self.is_baseexceptiongroup |= exc.is_baseexceptiongroup
            elif isinstance(exc, Matcher):
                if exc.exception_type is None:
                    continue
                self.is_baseexceptiongroup |= not issubclass(exc.exception_type, Exception)
            elif isinstance(exc, type) and issubclass(exc, BaseException):
                self.is_baseexceptiongroup |= not issubclass(exc, Exception)
            else:
                raise ValueError(f'Invalid argument "{exc!r}" must be exception type, Matcher, or RaisesGroup.')

    def __enter__(self) -> ExceptionInfo[BaseExceptionGroup[E]]:
        self.excinfo: ExceptionInfo[BaseExceptionGroup[E]] = ExceptionInfo.for_later()
        return self.excinfo

    def _unroll_exceptions(self, exceptions: Iterable[BaseException]) -> Iterable[BaseException]:
        """Used in non-strict mode."""
        res: list[BaseException] = []
        for exc in exceptions:
            if isinstance(exc, BaseExceptionGroup):
                res.extend(self._unroll_exceptions(exc.exceptions))
            else:
                res.append(exc)
        return res

    def matches(self, exc_val: BaseException | None) -> TypeGuard[BaseExceptionGroup[E]]:
        """Check if an exception matches the requirements of this RaisesGroup.

        Example::

            with pytest.raises(TypeError) as excinfo:
                ...
            assert RaisesGroups(ValueError).matches(excinfo.value.__cause__)
            # the above line is equivalent to
            myexc = excinfo.value.__cause
            assert isinstance(myexc, BaseExceptionGroup)
            assert len(myexc.exceptions) == 1
            assert isinstance(myexc.exceptions[0], ValueError)
        """
        if exc_val is None:
            return False
        if not isinstance(exc_val, BaseExceptionGroup):
            return False
        if len(exc_val.exceptions) != len(self.expected_exceptions):
            return False
        if self.match_expr is not None and (not re.search(self.match_expr, _stringify_exception(exc_val))):
            return False
        if self.check is not None and (not self.check(exc_val)):
            return False
        remaining_exceptions = list(self.expected_exceptions)
        actual_exceptions: Iterable[BaseException] = exc_val.exceptions
        if not self.strict:
            actual_exceptions = self._unroll_exceptions(actual_exceptions)
        for e in actual_exceptions:
            for rem_e in remaining_exceptions:
                if isinstance(rem_e, type) and isinstance(e, rem_e) or (isinstance(e, BaseExceptionGroup) and isinstance(rem_e, RaisesGroup) and rem_e.matches(e)) or (isinstance(rem_e, Matcher) and rem_e.matches(e)):
                    remaining_exceptions.remove(rem_e)
                    break
            else:
                return False
        return True

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> bool:
        __tracebackhide__ = True
        assert exc_type is not None, f'DID NOT RAISE any exception, expected {self.expected_type()}'
        assert self.excinfo is not None, 'Internal error - should have been constructed in __enter__'
        if not self.matches(exc_val):
            return False
        exc_info = cast('tuple[type[BaseExceptionGroup[E]], BaseExceptionGroup[E], types.TracebackType]', (exc_type, exc_val, exc_tb))
        self.excinfo.fill_unfilled(exc_info)
        return True

    def expected_type(self) -> str:
        subexcs = []
        for e in self.expected_exceptions:
            if isinstance(e, Matcher):
                subexcs.append(str(e))
            elif isinstance(e, RaisesGroup):
                subexcs.append(e.expected_type())
            elif isinstance(e, type):
                subexcs.append(e.__name__)
            else:
                raise AssertionError('unknown type')
        group_type = 'Base' if self.is_baseexceptiongroup else ''
        return f'{group_type}ExceptionGroup({', '.join(subexcs)})'