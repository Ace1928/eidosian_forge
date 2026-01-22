from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def _tarjan72(self):
    """Perform Tarjan's algorithm to find strongly connected components.

        See Also:
            :wikipedia:`Tarjan%27s_strongly_connected_components_algorithm`
        """
    result, stack, low = ([], [], {})

    def visit(node):
        if node in low:
            return
        num = len(low)
        low[node] = num
        stack_pos = len(stack)
        stack.append(node)
        for successor in self[node]:
            visit(successor)
            low[node] = min(low[node], low[successor])
        if num == low[node]:
            component = tuple(stack[stack_pos:])
            stack[stack_pos:] = []
            result.append(component)
            for item in component:
                low[item] = len(self)
    for node in self:
        visit(node)
    return result