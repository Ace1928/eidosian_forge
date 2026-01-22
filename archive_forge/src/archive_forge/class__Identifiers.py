import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
class _Identifiers:
    """tracks the status of identifier names as template code is rendered."""

    def __init__(self, compiler, node=None, parent=None, nested=False):
        if parent is not None:
            if isinstance(node, parsetree.NamespaceTag):
                self.declared = set()
                self.topleveldefs = util.SetLikeDict()
            else:
                self.declared = set(parent.declared).union([c.name for c in parent.closuredefs.values()]).union(parent.locally_declared).union(parent.argument_declared)
                if nested:
                    self.declared = self.declared.union(parent.undeclared)
                self.topleveldefs = util.SetLikeDict(**parent.topleveldefs)
        else:
            self.declared = set()
            self.topleveldefs = util.SetLikeDict()
        self.compiler = compiler
        self.undeclared = set()
        self.locally_declared = set()
        self.locally_assigned = set()
        self.argument_declared = set()
        self.closuredefs = util.SetLikeDict()
        self.node = node
        if node is not None:
            node.accept_visitor(self)
        illegal_names = self.compiler.reserved_names.intersection(self.locally_declared)
        if illegal_names:
            raise exceptions.NameConflictError('Reserved words declared in template: %s' % ', '.join(illegal_names))

    def branch(self, node, **kwargs):
        """create a new Identifiers for a new Node, with
        this Identifiers as the parent."""
        return _Identifiers(self.compiler, node, self, **kwargs)

    @property
    def defs(self):
        return set(self.topleveldefs.union(self.closuredefs).values())

    def __repr__(self):
        return 'Identifiers(declared=%r, locally_declared=%r, undeclared=%r, topleveldefs=%r, closuredefs=%r, argumentdeclared=%r)' % (list(self.declared), list(self.locally_declared), list(self.undeclared), [c.name for c in self.topleveldefs.values()], [c.name for c in self.closuredefs.values()], self.argument_declared)

    def check_declared(self, node):
        """update the state of this Identifiers with the undeclared
        and declared identifiers of the given node."""
        for ident in node.undeclared_identifiers():
            if ident != 'context' and ident not in self.declared.union(self.locally_declared):
                self.undeclared.add(ident)
        for ident in node.declared_identifiers():
            self.locally_declared.add(ident)

    def add_declared(self, ident):
        self.declared.add(ident)
        if ident in self.undeclared:
            self.undeclared.remove(ident)

    def visitExpression(self, node):
        self.check_declared(node)

    def visitControlLine(self, node):
        self.check_declared(node)

    def visitCode(self, node):
        if not node.ismodule:
            self.check_declared(node)
            self.locally_assigned = self.locally_assigned.union(node.declared_identifiers())

    def visitNamespaceTag(self, node):
        if self.node is node:
            for n in node.nodes:
                n.accept_visitor(self)

    def _check_name_exists(self, collection, node):
        existing = collection.get(node.funcname)
        collection[node.funcname] = node
        if existing is not None and existing is not node and (node.is_block or existing.is_block):
            raise exceptions.CompileException("%%def or %%block named '%s' already exists in this template." % node.funcname, **node.exception_kwargs)

    def visitDefTag(self, node):
        if node.is_root() and (not node.is_anonymous):
            self._check_name_exists(self.topleveldefs, node)
        elif node is not self.node:
            self._check_name_exists(self.closuredefs, node)
        for ident in node.undeclared_identifiers():
            if ident != 'context' and ident not in self.declared.union(self.locally_declared):
                self.undeclared.add(ident)
        if node is self.node:
            for ident in node.declared_identifiers():
                self.argument_declared.add(ident)
            for n in node.nodes:
                n.accept_visitor(self)

    def visitBlockTag(self, node):
        if node is not self.node and (not node.is_anonymous):
            if isinstance(self.node, parsetree.DefTag):
                raise exceptions.CompileException("Named block '%s' not allowed inside of def '%s'" % (node.name, self.node.name), **node.exception_kwargs)
            elif isinstance(self.node, (parsetree.CallTag, parsetree.CallNamespaceTag)):
                raise exceptions.CompileException("Named block '%s' not allowed inside of <%%call> tag" % (node.name,), **node.exception_kwargs)
        for ident in node.undeclared_identifiers():
            if ident != 'context' and ident not in self.declared.union(self.locally_declared):
                self.undeclared.add(ident)
        if not node.is_anonymous:
            self._check_name_exists(self.topleveldefs, node)
            self.undeclared.add(node.funcname)
        elif node is not self.node:
            self._check_name_exists(self.closuredefs, node)
        for ident in node.declared_identifiers():
            self.argument_declared.add(ident)
        for n in node.nodes:
            n.accept_visitor(self)

    def visitTextTag(self, node):
        for ident in node.undeclared_identifiers():
            if ident != 'context' and ident not in self.declared.union(self.locally_declared):
                self.undeclared.add(ident)

    def visitIncludeTag(self, node):
        self.check_declared(node)

    def visitPageTag(self, node):
        for ident in node.declared_identifiers():
            self.argument_declared.add(ident)
        self.check_declared(node)

    def visitCallNamespaceTag(self, node):
        self.visitCallTag(node)

    def visitCallTag(self, node):
        if node is self.node:
            for ident in node.undeclared_identifiers():
                if ident != 'context' and ident not in self.declared.union(self.locally_declared):
                    self.undeclared.add(ident)
            for ident in node.declared_identifiers():
                self.argument_declared.add(ident)
            for n in node.nodes:
                n.accept_visitor(self)
        else:
            for ident in node.undeclared_identifiers():
                if ident != 'context' and ident not in self.declared.union(self.locally_declared):
                    self.undeclared.add(ident)