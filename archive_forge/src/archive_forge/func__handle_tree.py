from functools import lru_cache
from traits.observation import _generated_parser
import traits.observation.expression as expression_module
def _handle_tree(tree, notify):
    """ Handle a tree using the specified rule.

    Parameters
    ----------
    tree : lark.tree.Tree
        Tree to be converted to an ObserverExpression.
    notify : bool
        True if the final target should notify, else False.

    Returns
    -------
    expression: ObserverExpression
    """
    handlers = {'series': _handle_series, 'series_terminal': _handle_series, 'parallel': _handle_parallel, 'parallel_terminal': _handle_parallel, 'trait': _handle_trait, 'metadata': _handle_metadata, 'items': _handle_items, 'anytrait': _handle_anytrait}
    return handlers[tree.data](tree.children, notify)