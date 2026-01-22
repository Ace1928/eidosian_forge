from __future__ import annotations
import warnings
from typing import cast
from typing import Generator
from typing import Mapping
from typing import NoReturn
from typing import Sequence
from typing import Tuple
from typing import Union
from ._hooks import HookImpl
from ._result import HookCallError
from ._result import Result
from ._warnings import PluggyTeardownRaisedWarning
def _multicall(hook_name: str, hook_impls: Sequence[HookImpl], caller_kwargs: Mapping[str, object], firstresult: bool) -> object | list[object]:
    """Execute a call into multiple python functions/methods and return the
    result(s).

    ``caller_kwargs`` comes from HookCaller.__call__().
    """
    __tracebackhide__ = True
    results: list[object] = []
    exception = None
    only_new_style_wrappers = True
    try:
        teardowns: list[Teardown] = []
        try:
            for hook_impl in reversed(hook_impls):
                try:
                    args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                except KeyError:
                    for argname in hook_impl.argnames:
                        if argname not in caller_kwargs:
                            raise HookCallError(f'hook call must provide argument {argname!r}')
                if hook_impl.hookwrapper:
                    only_new_style_wrappers = False
                    try:
                        res = hook_impl.function(*args)
                        wrapper_gen = cast(Generator[None, Result[object], None], res)
                        next(wrapper_gen)
                        teardowns.append((wrapper_gen, hook_impl))
                    except StopIteration:
                        _raise_wrapfail(wrapper_gen, 'did not yield')
                elif hook_impl.wrapper:
                    try:
                        res = hook_impl.function(*args)
                        function_gen = cast(Generator[None, object, object], res)
                        next(function_gen)
                        teardowns.append(function_gen)
                    except StopIteration:
                        _raise_wrapfail(function_gen, 'did not yield')
                else:
                    res = hook_impl.function(*args)
                    if res is not None:
                        results.append(res)
                        if firstresult:
                            break
        except BaseException as exc:
            exception = exc
    finally:
        if only_new_style_wrappers:
            if firstresult:
                result = results[0] if results else None
            else:
                result = results
            for teardown in reversed(teardowns):
                try:
                    if exception is not None:
                        teardown.throw(exception)
                    else:
                        teardown.send(result)
                    teardown.close()
                except StopIteration as si:
                    result = si.value
                    exception = None
                    continue
                except BaseException as e:
                    exception = e
                    continue
                _raise_wrapfail(teardown, 'has second yield')
            if exception is not None:
                raise exception.with_traceback(exception.__traceback__)
            else:
                return result
        else:
            if firstresult:
                outcome: Result[object | list[object]] = Result(results[0] if results else None, exception)
            else:
                outcome = Result(results, exception)
            for teardown in reversed(teardowns):
                if isinstance(teardown, tuple):
                    try:
                        teardown[0].send(outcome)
                    except StopIteration:
                        pass
                    except BaseException as e:
                        _warn_teardown_exception(hook_name, teardown[1], e)
                        raise
                    else:
                        _raise_wrapfail(teardown[0], 'has second yield')
                else:
                    try:
                        if outcome._exception is not None:
                            teardown.throw(outcome._exception)
                        else:
                            teardown.send(outcome._result)
                        teardown.close()
                    except StopIteration as si:
                        outcome.force_result(si.value)
                        continue
                    except BaseException as e:
                        outcome.force_exception(e)
                        continue
                    _raise_wrapfail(teardown, 'has second yield')
            return outcome.get_result()