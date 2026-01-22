important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class ImportName(Import):
    """For ``import_name`` nodes. Covers normal imports without ``from``."""
    type = 'import_name'
    __slots__ = ()

    def get_defined_names(self, include_setitem=False):
        """
        Returns the a list of `Name` that the import defines. The defined names
        is always the first name after `import` or in case an alias - `as` - is
        present that name is returned.
        """
        return [alias or path[0] for path, alias in self._dotted_as_names()]

    @property
    def level(self):
        """The level parameter of ``__import__``."""
        return 0

    def get_paths(self):
        return [path for path, alias in self._dotted_as_names()]

    def _dotted_as_names(self):
        """Generator of (list(path), alias) where alias may be None."""
        dotted_as_names = self.children[1]
        if dotted_as_names.type == 'dotted_as_names':
            as_names = dotted_as_names.children[::2]
        else:
            as_names = [dotted_as_names]
        for as_name in as_names:
            if as_name.type == 'dotted_as_name':
                alias = as_name.children[2]
                as_name = as_name.children[0]
            else:
                alias = None
            if as_name.type == 'name':
                yield ([as_name], alias)
            else:
                yield (as_name.children[::2], alias)

    def is_nested(self):
        """
        This checks for the special case of nested imports, without aliases and
        from statement::

            import foo.bar
        """
        return bool([1 for path, alias in self._dotted_as_names() if alias is None and len(path) > 1])

    def _aliases(self):
        """
        :return list of Name: Returns all the alias
        """
        return dict(((alias, path[-1]) for path, alias in self._dotted_as_names() if alias is not None))