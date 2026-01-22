from functools import lru_cache
from traits.observation import _generated_parser
import traits.observation.expression as expression_module
def _handle_series(trees, notify):
    """ Handle an expression of the form "a.b" or "a:b".

    Parameters
    ----------
    trees : list of lark.tree.Tree
        The children tree for the "series" rule. It should always
        contain exactly three items.
    notify : bool
        True if the final target should notify, else False.

    Returns
    -------
    expression : ObserverExpression
    """
    left, connector, right = trees
    notify_left = connector.data == 'notify'
    return _handle_tree(left, notify_left).then(_handle_tree(right, notify))