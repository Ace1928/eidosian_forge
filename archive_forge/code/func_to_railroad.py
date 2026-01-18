import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def to_railroad(element: pyparsing.ParserElement, diagram_kwargs: typing.Optional[dict]=None, vertical: int=3, show_results_names: bool=False, show_groups: bool=False) -> List[NamedDiagram]:
    """
    Convert a pyparsing element tree into a list of diagrams. This is the recommended entrypoint to diagram
    creation if you want to access the Railroad tree before it is converted to HTML
    :param element: base element of the parser being diagrammed
    :param diagram_kwargs: kwargs to pass to the Diagram() constructor
    :param vertical: (optional) - int - limit at which number of alternatives should be
       shown vertically instead of horizontally
    :param show_results_names - bool to indicate whether results name annotations should be
       included in the diagram
    :param show_groups - bool to indicate whether groups should be highlighted with an unlabeled
       surrounding box
    """
    lookup = ConverterState(diagram_kwargs=diagram_kwargs or {})
    _to_diagram_element(element, lookup=lookup, parent=None, vertical=vertical, show_results_names=show_results_names, show_groups=show_groups)
    root_id = id(element)
    if root_id in lookup:
        if not element.customName:
            lookup[root_id].name = ''
        lookup[root_id].mark_for_extraction(root_id, lookup, force=True)
    diags = list(lookup.diagrams.values())
    if len(diags) > 1:
        seen = set()
        deduped_diags = []
        for d in diags:
            if d.name == '...':
                continue
            if d.name is not None and d.name not in seen:
                seen.add(d.name)
                deduped_diags.append(d)
        resolved = [resolve_partial(partial) for partial in deduped_diags]
    else:
        resolved = [resolve_partial(partial) for partial in diags]
    return sorted(resolved, key=lambda diag: diag.index)