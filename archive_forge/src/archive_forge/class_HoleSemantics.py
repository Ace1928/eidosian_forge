from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
class HoleSemantics:
    """
    This class holds the broken-down components of a hole semantics, i.e. it
    extracts the holes, labels, logic formula fragments and constraints out of
    a big conjunction of such as produced by the hole semantics grammar.  It
    then provides some operations on the semantics dealing with holes, labels
    and finding legal ways to plug holes with labels.
    """

    def __init__(self, usr):
        """
        Constructor.  `usr' is a ``sem.Expression`` representing an
        Underspecified Representation Structure (USR).  A USR has the following
        special predicates:
        ALL(l,v,n),
        EXISTS(l,v,n),
        AND(l,n,n),
        OR(l,n,n),
        IMP(l,n,n),
        IFF(l,n,n),
        PRED(l,v,n,v[,v]*) where the brackets and star indicate zero or more repetitions,
        LEQ(n,n),
        HOLE(n),
        LABEL(n)
        where l is the label of the node described by the predicate, n is either
        a label or a hole, and v is a variable.
        """
        self.holes = set()
        self.labels = set()
        self.fragments = {}
        self.constraints = set()
        self._break_down(usr)
        self.top_most_labels = self._find_top_most_labels()
        self.top_hole = self._find_top_hole()

    def is_node(self, x):
        """
        Return true if x is a node (label or hole) in this semantic
        representation.
        """
        return x in self.labels | self.holes

    def _break_down(self, usr):
        """
        Extract holes, labels, formula fragments and constraints from the hole
        semantics underspecified representation (USR).
        """
        if isinstance(usr, AndExpression):
            self._break_down(usr.first)
            self._break_down(usr.second)
        elif isinstance(usr, ApplicationExpression):
            func, args = usr.uncurry()
            if func.variable.name == Constants.LEQ:
                self.constraints.add(Constraint(args[0], args[1]))
            elif func.variable.name == Constants.HOLE:
                self.holes.add(args[0])
            elif func.variable.name == Constants.LABEL:
                self.labels.add(args[0])
            else:
                label = args[0]
                assert label not in self.fragments
                self.fragments[label] = (func, args[1:])
        else:
            raise ValueError(usr.label())

    def _find_top_nodes(self, node_list):
        top_nodes = node_list.copy()
        for f in self.fragments.values():
            args = f[1]
            for arg in args:
                if arg in node_list:
                    top_nodes.discard(arg)
        return top_nodes

    def _find_top_most_labels(self):
        """
        Return the set of labels which are not referenced directly as part of
        another formula fragment.  These will be the top-most labels for the
        subtree that they are part of.
        """
        return self._find_top_nodes(self.labels)

    def _find_top_hole(self):
        """
        Return the hole that will be the top of the formula tree.
        """
        top_holes = self._find_top_nodes(self.holes)
        assert len(top_holes) == 1
        return top_holes.pop()

    def pluggings(self):
        """
        Calculate and return all the legal pluggings (mappings of labels to
        holes) of this semantics given the constraints.
        """
        record = []
        self._plug_nodes([(self.top_hole, [])], self.top_most_labels, {}, record)
        return record

    def _plug_nodes(self, queue, potential_labels, plug_acc, record):
        """
        Plug the nodes in `queue' with the labels in `potential_labels'.

        Each element of `queue' is a tuple of the node to plug and the list of
        ancestor holes from the root of the graph to that node.

        `potential_labels' is a set of the labels which are still available for
        plugging.

        `plug_acc' is the incomplete mapping of holes to labels made on the
        current branch of the search tree so far.

        `record' is a list of all the complete pluggings that we have found in
        total so far.  It is the only parameter that is destructively updated.
        """
        if queue != []:
            node, ancestors = queue[0]
            if node in self.holes:
                self._plug_hole(node, ancestors, queue[1:], potential_labels, plug_acc, record)
            else:
                assert node in self.labels
                args = self.fragments[node][1]
                head = [(a, ancestors) for a in args if self.is_node(a)]
                self._plug_nodes(head + queue[1:], potential_labels, plug_acc, record)
        else:
            raise Exception('queue empty')

    def _plug_hole(self, hole, ancestors0, queue, potential_labels0, plug_acc0, record):
        """
        Try all possible ways of plugging a single hole.
        See _plug_nodes for the meanings of the parameters.
        """
        assert hole not in ancestors0
        ancestors = [hole] + ancestors0
        for l in potential_labels0:
            if self._violates_constraints(l, ancestors):
                continue
            plug_acc = plug_acc0.copy()
            plug_acc[hole] = l
            potential_labels = potential_labels0.copy()
            potential_labels.remove(l)
            if len(potential_labels) == 0:
                self._sanity_check_plugging(plug_acc, self.top_hole, [])
                record.append(plug_acc)
            else:
                self._plug_nodes(queue + [(l, ancestors)], potential_labels, plug_acc, record)

    def _violates_constraints(self, label, ancestors):
        """
        Return True if the `label' cannot be placed underneath the holes given
        by the set `ancestors' because it would violate the constraints imposed
        on it.
        """
        for c in self.constraints:
            if c.lhs == label:
                if c.rhs not in ancestors:
                    return True
        return False

    def _sanity_check_plugging(self, plugging, node, ancestors):
        """
        Make sure that a given plugging is legal.  We recursively go through
        each node and make sure that no constraints are violated.
        We also check that all holes have been filled.
        """
        if node in self.holes:
            ancestors = [node] + ancestors
            label = plugging[node]
        else:
            label = node
        assert label in self.labels
        for c in self.constraints:
            if c.lhs == label:
                assert c.rhs in ancestors
        args = self.fragments[label][1]
        for arg in args:
            if self.is_node(arg):
                self._sanity_check_plugging(plugging, arg, [label] + ancestors)

    def formula_tree(self, plugging):
        """
        Return the first-order logic formula tree for this underspecified
        representation using the plugging given.
        """
        return self._formula_tree(plugging, self.top_hole)

    def _formula_tree(self, plugging, node):
        if node in plugging:
            return self._formula_tree(plugging, plugging[node])
        elif node in self.fragments:
            pred, args = self.fragments[node]
            children = [self._formula_tree(plugging, arg) for arg in args]
            return reduce(Constants.MAP[pred.variable.name], children)
        else:
            return node