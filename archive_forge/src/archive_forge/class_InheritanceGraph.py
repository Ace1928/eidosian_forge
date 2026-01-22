import builtins
import inspect
import re
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.graphviz import (figure_wrapper, graphviz, render_dot_html, render_dot_latex,
from sphinx.util import md5
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.texinfo import TexinfoTranslator
class InheritanceGraph:
    """
    Given a list of classes, determines the set of classes that they inherit
    from all the way to the root "object", and then is able to generate a
    graphviz dot graph from them.
    """

    def __init__(self, class_names: List[str], currmodule: str, show_builtins: bool=False, private_bases: bool=False, parts: int=0, aliases: Optional[Dict[str, str]]=None, top_classes: List[Any]=[]) -> None:
        """*class_names* is a list of child classes to show bases from.

        If *show_builtins* is True, then Python builtins will be shown
        in the graph.
        """
        self.class_names = class_names
        classes = self._import_classes(class_names, currmodule)
        self.class_info = self._class_info(classes, show_builtins, private_bases, parts, aliases, top_classes)
        if not self.class_info:
            raise InheritanceException('No classes found for inheritance diagram')

    def _import_classes(self, class_names: List[str], currmodule: str) -> List[Any]:
        """Import a list of classes."""
        classes: List[Any] = []
        for name in class_names:
            classes.extend(import_classes(name, currmodule))
        return classes

    def _class_info(self, classes: List[Any], show_builtins: bool, private_bases: bool, parts: int, aliases: Dict[str, str], top_classes: List[Any]) -> List[Tuple[str, str, List[str], str]]:
        """Return name and bases for all classes that are ancestors of
        *classes*.

        *parts* gives the number of dotted name parts to include in the
        displayed node names, from right to left. If given as a negative, the
        number of parts to drop from the left. A value of 0 displays the full
        dotted name. E.g. ``sphinx.ext.inheritance_diagram.InheritanceGraph``
        with ``parts=2`` or ``parts=-2`` gets displayed as
        ``inheritance_diagram.InheritanceGraph``, and as
        ``ext.inheritance_diagram.InheritanceGraph`` with ``parts=3`` or
        ``parts=-1``.

        *top_classes* gives the name(s) of the top most ancestor class to
        traverse to. Multiple names can be specified separated by comma.
        """
        all_classes = {}

        def recurse(cls: Any) -> None:
            if not show_builtins and cls in py_builtins:
                return
            if not private_bases and cls.__name__.startswith('_'):
                return
            nodename = self.class_name(cls, parts, aliases)
            fullname = self.class_name(cls, 0, aliases)
            tooltip = None
            try:
                if cls.__doc__:
                    doc = cls.__doc__.strip().split('\n')[0]
                    if doc:
                        tooltip = '"%s"' % doc.replace('"', '\\"')
            except Exception:
                pass
            baselist: List[str] = []
            all_classes[cls] = (nodename, fullname, baselist, tooltip)
            if fullname in top_classes:
                return
            for base in cls.__bases__:
                if not show_builtins and base in py_builtins:
                    continue
                if not private_bases and base.__name__.startswith('_'):
                    continue
                baselist.append(self.class_name(base, parts, aliases))
                if base not in all_classes:
                    recurse(base)
        for cls in classes:
            recurse(cls)
        return list(all_classes.values())

    def class_name(self, cls: Any, parts: int=0, aliases: Optional[Dict[str, str]]=None) -> str:
        """Given a class object, return a fully-qualified name.

        This works for things I've tested in matplotlib so far, but may not be
        completely general.
        """
        module = cls.__module__
        if module in ('__builtin__', 'builtins'):
            fullname = cls.__name__
        else:
            fullname = '%s.%s' % (module, cls.__qualname__)
        if parts == 0:
            result = fullname
        else:
            name_parts = fullname.split('.')
            result = '.'.join(name_parts[-parts:])
        if aliases is not None and result in aliases:
            return aliases[result]
        return result

    def get_all_class_names(self) -> List[str]:
        """Get all of the class names involved in the graph."""
        return [fullname for _, fullname, _, _ in self.class_info]
    default_graph_attrs = {'rankdir': 'LR', 'size': '"8.0, 12.0"', 'bgcolor': 'transparent'}
    default_node_attrs = {'shape': 'box', 'fontsize': 10, 'height': 0.25, 'fontname': '"Vera Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica, sans"', 'style': '"setlinewidth(0.5),filled"', 'fillcolor': 'white'}
    default_edge_attrs = {'arrowsize': 0.5, 'style': '"setlinewidth(0.5)"'}

    def _format_node_attrs(self, attrs: Dict[str, Any]) -> str:
        return ','.join(['%s=%s' % x for x in sorted(attrs.items())])

    def _format_graph_attrs(self, attrs: Dict[str, Any]) -> str:
        return ''.join(['%s=%s;\n' % x for x in sorted(attrs.items())])

    def generate_dot(self, name: str, urls: Dict[str, str]={}, env: Optional[BuildEnvironment]=None, graph_attrs: Dict={}, node_attrs: Dict={}, edge_attrs: Dict={}) -> str:
        """Generate a graphviz dot graph from the classes that were passed in
        to __init__.

        *name* is the name of the graph.

        *urls* is a dictionary mapping class names to HTTP URLs.

        *graph_attrs*, *node_attrs*, *edge_attrs* are dictionaries containing
        key/value pairs to pass on as graphviz properties.
        """
        g_attrs = self.default_graph_attrs.copy()
        n_attrs = self.default_node_attrs.copy()
        e_attrs = self.default_edge_attrs.copy()
        g_attrs.update(graph_attrs)
        n_attrs.update(node_attrs)
        e_attrs.update(edge_attrs)
        if env:
            g_attrs.update(env.config.inheritance_graph_attrs)
            n_attrs.update(env.config.inheritance_node_attrs)
            e_attrs.update(env.config.inheritance_edge_attrs)
        res: List[str] = []
        res.append('digraph %s {\n' % name)
        res.append(self._format_graph_attrs(g_attrs))
        for name, fullname, bases, tooltip in sorted(self.class_info):
            this_node_attrs = n_attrs.copy()
            if fullname in urls:
                this_node_attrs['URL'] = '"%s"' % urls[fullname]
                this_node_attrs['target'] = '"_top"'
            if tooltip:
                this_node_attrs['tooltip'] = tooltip
            res.append('  "%s" [%s];\n' % (name, self._format_node_attrs(this_node_attrs)))
            for base_name in bases:
                res.append('  "%s" -> "%s" [%s];\n' % (base_name, name, self._format_node_attrs(e_attrs)))
        res.append('}\n')
        return ''.join(res)