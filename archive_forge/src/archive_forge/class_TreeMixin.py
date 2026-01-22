import collections
import copy
import itertools
import random
import re
import warnings
class TreeMixin:
    """Methods for Tree- and Clade-based classes.

    This lets ``Tree`` and ``Clade`` support the same traversal and searching
    operations without requiring Clade to inherit from Tree, so Clade isn't
    required to have all of Tree's attributes -- just ``root`` (a Clade
    instance) and ``is_terminal``.
    """

    def _filter_search(self, filter_func, order, follow_attrs):
        """Perform a BFS or DFS traversal through all elements in this tree (PRIVATE).

        :returns: generator of all elements for which ``filter_func`` is True.

        """
        order_opts = {'preorder': _preorder_traverse, 'postorder': _postorder_traverse, 'level': _level_traverse}
        try:
            order_func = order_opts[order]
        except KeyError:
            raise ValueError(f"Invalid order '{order}'; must be one of: {tuple(order_opts)}") from None
        if follow_attrs:
            get_children = _sorted_attrs
            root = self
        else:
            get_children = lambda elem: elem.clades
            root = self.root
        return filter(filter_func, order_func(root, get_children))

    def find_any(self, *args, **kwargs):
        """Return the first element found by find_elements(), or None.

        This is also useful for checking whether any matching element exists in
        the tree, and can be used in a conditional expression.
        """
        hits = self.find_elements(*args, **kwargs)
        try:
            return next(hits)
        except StopIteration:
            return None

    def find_elements(self, target=None, terminal=None, order='preorder', **kwargs):
        """Find all tree elements matching the given attributes.

        The arbitrary keyword arguments indicate the attribute name of the
        sub-element and the value to match: string, integer or boolean. Strings
        are evaluated as regular expression matches; integers are compared
        directly for equality, and booleans evaluate the attribute's truth value
        (True or False) before comparing. To handle nonzero floats, search with
        a boolean argument, then filter the result manually.

        If no keyword arguments are given, then just the class type is used for
        matching.

        The result is an iterable through all matching objects, by depth-first
        search. (Not necessarily the same order as the elements appear in the
        source file!)

        :Parameters:
            target : TreeElement instance, type, dict, or callable
                Specifies the characteristics to search for. (The default,
                TreeElement, matches any standard Bio.Phylo type.)
            terminal : bool
                A boolean value to select for or against terminal nodes (a.k.a.
                leaf nodes). True searches for only terminal nodes, False
                excludes terminal nodes, and the default, None, searches both
                terminal and non-terminal nodes, as well as any tree elements
                lacking the ``is_terminal`` method.
            order : {'preorder', 'postorder', 'level'}
                Tree traversal order: 'preorder' (default) is depth-first
                search, 'postorder' is DFS with child nodes preceding parents,
                and 'level' is breadth-first search.

        Examples
        --------
        >>> from Bio import Phylo
        >>> phx = Phylo.PhyloXMLIO.read('PhyloXML/phyloxml_examples.xml')
        >>> matches = phx.phylogenies[5].find_elements(code='OCTVU')
        >>> next(matches)
        Taxonomy(code='OCTVU', scientific_name='Octopus vulgaris')

        """
        if terminal is not None:
            kwargs['terminal'] = terminal
        is_matching_elem = _combine_matchers(target, kwargs, False)
        return self._filter_search(is_matching_elem, order, True)

    def find_clades(self, target=None, terminal=None, order='preorder', **kwargs):
        """Find each clade containing a matching element.

        That is, find each element as with find_elements(), but return the
        corresponding clade object. (This is usually what you want.)

        :returns: an iterable through all matching objects, searching
            depth-first (preorder) by default.

        """

        def match_attrs(elem):
            orig_clades = elem.__dict__.pop('clades')
            found = elem.find_any(target, **kwargs)
            elem.clades = orig_clades
            return found is not None
        if terminal is None:
            is_matching_elem = match_attrs
        else:

            def is_matching_elem(elem):
                return elem.is_terminal() == terminal and match_attrs(elem)
        return self._filter_search(is_matching_elem, order, False)

    def get_path(self, target=None, **kwargs):
        """List the clades directly between this root and the given target.

        :returns: list of all clade objects along this path, ending with the
            given target, but excluding the root clade.

        """
        path = []
        match = _combine_matchers(target, kwargs, True)

        def check_in_path(v):
            if match(v):
                path.append(v)
                return True
            elif v.is_terminal():
                return False
            for child in v:
                if check_in_path(child):
                    path.append(v)
                    return True
            return False
        if not check_in_path(self.root):
            return None
        return path[-2::-1]

    def get_nonterminals(self, order='preorder'):
        """Get a list of all of this tree's nonterminal (internal) nodes."""
        return list(self.find_clades(terminal=False, order=order))

    def get_terminals(self, order='preorder'):
        """Get a list of all of this tree's terminal (leaf) nodes."""
        return list(self.find_clades(terminal=True, order=order))

    def trace(self, start, finish):
        """List of all clade object between two targets in this tree.

        Excluding ``start``, including ``finish``.
        """
        mrca = self.common_ancestor(start, finish)
        fromstart = mrca.get_path(start)[-2::-1]
        to = mrca.get_path(finish)
        return fromstart + [mrca] + to

    def common_ancestor(self, targets, *more_targets):
        """Most recent common ancestor (clade) of all the given targets.

        Edge cases:
         - If no target is given, returns self.root
         - If 1 target is given, returns the target
         - If any target is not found in this tree, raises a ValueError

        """
        paths = [self.get_path(t) for t in _combine_args(targets, *more_targets)]
        for p, t in zip(paths, targets):
            if p is None:
                raise ValueError(f'target {t!r} is not in this tree')
        mrca = self.root
        for level in zip(*paths):
            ref = level[0]
            for other in level[1:]:
                if ref is not other:
                    break
            else:
                mrca = ref
            if ref is not mrca:
                break
        return mrca

    def count_terminals(self):
        """Count the number of terminal (leaf) nodes within this tree."""
        return sum((1 for clade in self.find_clades(terminal=True)))

    def depths(self, unit_branch_lengths=False):
        """Create a mapping of tree clades to depths (by branch length).

        :Parameters:
            unit_branch_lengths : bool
                If True, count only the number of branches (levels in the tree).
                By default the distance is the cumulative branch length leading
                to the clade.

        :returns: dict of {clade: depth}, where keys are all of the Clade
            instances in the tree, and values are the distance from the root to
            each clade (including terminals).

        """
        if unit_branch_lengths:
            depth_of = lambda c: 1
        else:
            depth_of = lambda c: c.branch_length or 0
        depths = {}

        def update_depths(node, curr_depth):
            depths[node] = curr_depth
            for child in node.clades:
                new_depth = curr_depth + depth_of(child)
                update_depths(child, new_depth)
        update_depths(self.root, self.root.branch_length or 0)
        return depths

    def distance(self, target1, target2=None):
        """Calculate the sum of the branch lengths between two targets.

        If only one target is specified, the other is the root of this tree.
        """
        if target2 is None:
            return sum((n.branch_length for n in self.get_path(target1) if n.branch_length is not None))
        mrca = self.common_ancestor(target1, target2)
        return mrca.distance(target1) + mrca.distance(target2)

    def is_bifurcating(self):
        """Return True if tree downstream of node is strictly bifurcating.

        I.e., all nodes have either 2 or 0 children (internal or external,
        respectively). The root may have 3 descendents and still be considered
        part of a bifurcating tree, because it has no ancestor.
        """
        if isinstance(self, Tree) and len(self.root) == 3:
            return self.root.clades[0].is_bifurcating() and self.root.clades[1].is_bifurcating() and self.root.clades[2].is_bifurcating()
        if len(self.root) == 2:
            return self.root.clades[0].is_bifurcating() and self.root.clades[1].is_bifurcating()
        if len(self.root) == 0:
            return True
        return False

    def is_monophyletic(self, terminals, *more_terminals):
        """MRCA of terminals if they comprise a complete subclade, or False.

        I.e., there exists a clade such that its terminals are the same set as
        the given targets.

        The given targets must be terminals of the tree.

        To match both ``Bio.Nexus.Trees`` and the other multi-target methods in
        Bio.Phylo, arguments to this method can be specified either of two ways:
        (i) as a single list of targets, or (ii) separately specified targets,
        e.g. is_monophyletic(t1, t2, t3) -- but not both.

        For convenience, this method returns the common ancestor (MCRA) of the
        targets if they are monophyletic (instead of the value True), and False
        otherwise.

        :returns: common ancestor if terminals are monophyletic, otherwise False.

        """
        target_set = set(_combine_args(terminals, *more_terminals))
        current = self.root
        while True:
            if set(current.get_terminals()) == target_set:
                return current
            for subclade in current.clades:
                if set(subclade.get_terminals()).issuperset(target_set):
                    current = subclade
                    break
            else:
                return False

    def is_parent_of(self, target=None, **kwargs):
        """Check if target is a descendent of this tree.

        Not required to be a direct descendent.

        To check only direct descendents of a clade, simply use list membership
        testing: ``if subclade in clade: ...``
        """
        return self.get_path(target, **kwargs) is not None

    def is_preterminal(self):
        """Check if all direct descendents are terminal."""
        if self.root.is_terminal():
            return False
        for clade in self.root.clades:
            if not clade.is_terminal():
                return False
        return True

    def total_branch_length(self):
        """Calculate the sum of all the branch lengths in this tree."""
        return sum((node.branch_length for node in self.find_clades(branch_length=True)))

    def collapse(self, target=None, **kwargs):
        """Delete target from the tree, relinking its children to its parent.

        :returns: the parent clade.

        """
        path = self.get_path(target, **kwargs)
        if not path:
            raise ValueError("couldn't collapse %s in this tree" % (target or kwargs))
        if len(path) == 1:
            parent = self.root
        else:
            parent = path[-2]
        popped = parent.clades.pop(parent.clades.index(path[-1]))
        extra_length = popped.branch_length or 0
        for child in popped:
            child.branch_length += extra_length
        parent.clades.extend(popped.clades)
        return parent

    def collapse_all(self, target=None, **kwargs):
        """Collapse all the descendents of this tree, leaving only terminals.

        Total branch lengths are preserved, i.e. the distance to each terminal
        stays the same.

        For example, this will safely collapse nodes with poor bootstrap
        support:

            >>> from Bio import Phylo
            >>> tree = Phylo.read('PhyloXML/apaf.xml', 'phyloxml')
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 20.44
            >>> tree.collapse_all(lambda c: c.confidence is not None and c.confidence < 70)
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 21.37

        This implementation avoids strange side-effects by using level-order
        traversal and testing all clade properties (versus the target
        specification) up front. In particular, if a clade meets the target
        specification in the original tree, it will be collapsed.  For example,
        if the condition is:

            >>> from Bio import Phylo
            >>> tree = Phylo.read('PhyloXML/apaf.xml', 'phyloxml')
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 20.44
            >>> tree.collapse_all(lambda c: c.branch_length < 0.1)
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 21.13

        Collapsing a clade's parent node adds the parent's branch length to the
        child, so during the execution of collapse_all, a clade's branch_length
        may increase. In this implementation, clades are collapsed according to
        their properties in the original tree, not the properties when tree
        traversal reaches the clade. (It's easier to debug.) If you want the
        other behavior (incremental testing), modifying the source code of this
        function is straightforward.
        """
        matches = list(self.find_clades(target, False, 'level', **kwargs))
        if not matches:
            return
        if matches[0] == self.root:
            matches.pop(0)
        for clade in matches:
            self.collapse(clade)

    def ladderize(self, reverse=False):
        """Sort clades in-place according to the number of terminal nodes.

        Deepest clades are last by default. Use ``reverse=True`` to sort clades
        deepest-to-shallowest.
        """
        self.root.clades.sort(key=lambda c: c.count_terminals(), reverse=reverse)
        for subclade in self.root.clades:
            subclade.ladderize(reverse=reverse)

    def prune(self, target=None, **kwargs):
        """Prunes a terminal clade from the tree.

        If taxon is from a bifurcation, the connecting node will be collapsed
        and its branch length added to remaining terminal node. This might be no
        longer be a meaningful value.

        :returns: parent clade of the pruned target

        """
        if 'terminal' in kwargs and kwargs['terminal']:
            raise ValueError('target must be terminal')
        path = self.get_path(target, terminal=True, **kwargs)
        if not path:
            raise ValueError("can't find a matching target below this root")
        if len(path) == 1:
            parent = self.root
        else:
            parent = path[-2]
        parent.clades.remove(path[-1])
        if len(parent) == 1:
            if parent == self.root:
                newroot = parent.clades[0]
                newroot.branch_length = None
                parent = self.root = newroot
            else:
                child = parent.clades[0]
                if child.branch_length is not None:
                    child.branch_length += parent.branch_length or 0.0
                if len(path) < 3:
                    grandparent = self.root
                else:
                    grandparent = path[-3]
                index = grandparent.clades.index(parent)
                grandparent.clades.pop(index)
                grandparent.clades.insert(index, child)
                parent = grandparent
        return parent

    def split(self, n=2, branch_length=1.0):
        """Generate n (default 2) new descendants.

        In a species tree, this is a speciation event.

        New clades have the given branch_length and the same name as this
        clade's root plus an integer suffix (counting from 0). For example,
        splitting a clade named "A" produces sub-clades named "A0" and "A1".
        If the clade has no name, the prefix "n" is used for child nodes, e.g.
        "n0" and "n1".
        """
        clade_cls = type(self.root)
        base_name = self.root.name or 'n'
        for i in range(n):
            clade = clade_cls(name=base_name + str(i), branch_length=branch_length)
            self.root.clades.append(clade)