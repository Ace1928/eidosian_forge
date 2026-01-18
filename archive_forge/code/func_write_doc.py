from __future__ import annotations
import typing as t
from collections import defaultdict
from textwrap import dedent
from traitlets import HasTraits, Undefined
from traitlets.config.application import Application
from traitlets.utils.text import indent
def write_doc(path: str, title: str, app: Application, preamble: str | None=None) -> None:
    """Write a rst file documenting config options for a traitlets application.

    Parameters
    ----------
    path : str
        The file to be written
    title : str
        The human-readable title of the document
    app : traitlets.config.Application
        An instance of the application class to be documented
    preamble : str
        Extra text to add just after the title (optional)
    """
    trait_aliases = reverse_aliases(app)
    with open(path, 'w') as f:
        f.write(title + '\n')
        f.write('=' * len(title) + '\n')
        f.write('\n')
        if preamble is not None:
            f.write(preamble + '\n\n')
        for c in app._classes_inc_parents():
            f.write(class_config_rst_doc(c, trait_aliases))
            f.write('\n')