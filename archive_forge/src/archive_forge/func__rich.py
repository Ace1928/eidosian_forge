import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def _rich(self, parent):
    if parent:
        tree = parent.add(f'[bold]{self.data}[/bold]')
    else:
        import rich.tree
        tree = rich.tree.Tree(self.data)
    for c in self.children:
        if isinstance(c, Tree):
            c._rich(tree)
        else:
            tree.add(f'[green]{c}[/green]')
    return tree