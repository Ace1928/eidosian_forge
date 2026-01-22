import inspect
from contextlib import contextmanager
from typing import Any, Optional, Set, Tuple
import ray  # noqa: F401
import colorama
import ray.cloudpickle as cp
from ray.util.annotations import DeveloperAPI
def _inspect_serializability(base_obj, name, depth, parent, failure_set, printer) -> Tuple[bool, Set[FailureTuple]]:
    colorama.init()
    top_level = False
    declaration = ''
    found = False
    if failure_set is None:
        top_level = True
        failure_set = set()
        declaration = f'Checking Serializability of {base_obj}'
        printer.print('=' * min(len(declaration), 80))
        printer.print(declaration)
        printer.print('=' * min(len(declaration), 80))
        if name is None:
            name = str(base_obj)
    else:
        printer.print(f"Serializing '{name}' {base_obj}...")
    try:
        cp.dumps(base_obj)
        return (True, failure_set)
    except Exception as e:
        printer.print(f'{colorama.Fore.RED}!!! FAIL{colorama.Fore.RESET} serialization: {e}')
        found = True
        try:
            if depth == 0:
                failure_set.add(FailureTuple(base_obj, name, parent))
        except Exception:
            pass
    if depth <= 0:
        return (False, failure_set)
    if inspect.isfunction(base_obj):
        _inspect_func_serialization(base_obj, depth=depth, parent=base_obj, failure_set=failure_set, printer=printer)
    else:
        _inspect_generic_serialization(base_obj, depth=depth, parent=base_obj, failure_set=failure_set, printer=printer)
    if not failure_set:
        failure_set.add(FailureTuple(base_obj, name, parent))
    if top_level:
        printer.print('=' * min(len(declaration), 80))
        if not failure_set:
            printer.print('Nothing failed the inspect_serialization test, though serialization did not succeed.')
        else:
            fail_vars = f'\n\n\t{colorama.Style.BRIGHT}' + '\n'.join((str(k) for k in failure_set)) + f'{colorama.Style.RESET_ALL}\n\n'
            printer.print(f'Variable: {fail_vars}was found to be non-serializable. There may be multiple other undetected variables that were non-serializable. ')
            printer.print('Consider either removing the instantiation/imports of these variables or moving the instantiation into the scope of the function/class. ')
        printer.print('=' * min(len(declaration), 80))
        printer.print('Check https://docs.ray.io/en/master/ray-core/objects/serialization.html#troubleshooting for more information.')
        printer.print('If you have any suggestions on how to improve this error message, please reach out to the Ray developers on github.com/ray-project/ray/issues/')
        printer.print('=' * min(len(declaration), 80))
    return (not found, failure_set)