from __future__ import annotations
import json
import logging
import typing as t
from dataclasses import dataclass, field
from sqlglot import Schema, exp, maybe_parse
from sqlglot.errors import SqlglotError
from sqlglot.optimizer import Scope, build_scope, find_all_in_scope, normalize_identifiers, qualify
class GraphHTML:
    """Node to HTML generator using vis.js.

    https://visjs.github.io/vis-network/docs/network/
    """

    def __init__(self, nodes: t.Dict, edges: t.List, imports: bool=True, options: t.Optional[t.Dict]=None):
        self.imports = imports
        self.options = {'height': '500px', 'width': '100%', 'layout': {'hierarchical': {'enabled': True, 'nodeSpacing': 200, 'sortMethod': 'directed'}}, 'interaction': {'dragNodes': False, 'selectable': False}, 'physics': {'enabled': False}, 'edges': {'arrows': 'to'}, 'nodes': {'font': '20px monaco', 'shape': 'box', 'widthConstraint': {'maximum': 300}}, **(options or {})}
        self.nodes = nodes
        self.edges = edges

    def __str__(self):
        nodes = json.dumps(list(self.nodes.values()))
        edges = json.dumps(self.edges)
        options = json.dumps(self.options)
        imports = '<script type="text/javascript" src="https://unpkg.com/vis-data@latest/peer/umd/vis-data.min.js"></script>\n  <script type="text/javascript" src="https://unpkg.com/vis-network@latest/peer/umd/vis-network.min.js"></script>\n  <link rel="stylesheet" type="text/css" href="https://unpkg.com/vis-network/styles/vis-network.min.css" />' if self.imports else ''
        return f'<div>\n  <div id="sqlglot-lineage"></div>\n  {imports}\n  <script type="text/javascript">\n    var nodes = new vis.DataSet({nodes})\n    nodes.forEach(row => row["title"] = new DOMParser().parseFromString(row["title"], "text/html").body.childNodes[0])\n\n    new vis.Network(\n        document.getElementById("sqlglot-lineage"),\n        {{\n            nodes: nodes,\n            edges: new vis.DataSet({edges})\n        }},\n        {options},\n    )\n  </script>\n</div>'

    def _repr_html_(self) -> str:
        return self.__str__()