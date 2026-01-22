from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class HTMLTemplateFormatter(CellFormatter):
    """ HTML formatter using a template.
    This uses Underscore's `template` method and syntax. http://underscorejs.org/#template
    The formatter has access other items in the row via the `dataContext` object passed to the formatter.
    So, for example, if another column in the datasource was named `url`, the template could access it as:

    .. code-block:: jinja

        <a href="<%= url %>"><%= value %></a>

    To use a different set of template delimiters, pass the appropriate values for `evaluate`, `interpolate`,
    or `escape`. See the Underscore `template` documentation for more information. http://underscorejs.org/#template

    Example: Simple HTML template to format the column value as code.

    .. code-block:: python

        HTMLTemplateFormatter(template='<code><%= value %></code>')

    Example: Use values from other columns (`manufacturer` and `model`) to build a hyperlink.

    .. code-block:: python

        HTMLTemplateFormatter(template=
            '<a href="https:/www.google.com/search?q=<%= manufacturer %>+<%= model %>" target="_blank"><%= value %></a>'
        )

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    template = String('<%= value %>', help="\n    Template string to be used by Underscore's template method.\n    ")