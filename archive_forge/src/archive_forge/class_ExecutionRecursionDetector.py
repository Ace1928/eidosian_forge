from contextlib import contextmanager
from jedi import debug
from jedi.inference.base_value import NO_VALUES
class ExecutionRecursionDetector:
    """
    Catches recursions of executions.
    """

    def __init__(self, inference_state):
        self._inference_state = inference_state
        self._recursion_level = 0
        self._parent_execution_funcs = []
        self._funcdef_execution_counts = {}
        self._execution_count = 0

    def pop_execution(self):
        self._parent_execution_funcs.pop()
        self._recursion_level -= 1

    def push_execution(self, execution):
        funcdef = execution.tree_node
        self._recursion_level += 1
        self._parent_execution_funcs.append(funcdef)
        module_context = execution.get_root_context()
        if module_context.is_builtins_module():
            return False
        if self._recursion_level > recursion_limit:
            debug.warning('Recursion limit (%s) reached', recursion_limit)
            return True
        if self._execution_count >= total_function_execution_limit:
            debug.warning('Function execution limit (%s) reached', total_function_execution_limit)
            return True
        self._execution_count += 1
        if self._funcdef_execution_counts.setdefault(funcdef, 0) >= per_function_execution_limit:
            if module_context.py__name__() == 'typing':
                return False
            debug.warning('Per function execution limit (%s) reached: %s', per_function_execution_limit, funcdef)
            return True
        self._funcdef_execution_counts[funcdef] += 1
        if self._parent_execution_funcs.count(funcdef) > per_function_recursion_limit:
            debug.warning('Per function recursion limit (%s) reached: %s', per_function_recursion_limit, funcdef)
            return True
        return False