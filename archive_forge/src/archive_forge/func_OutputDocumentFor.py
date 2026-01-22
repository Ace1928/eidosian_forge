from __future__ import annotations
import logging # isort:skip
import re
from contextlib import contextmanager
from typing import (
from weakref import WeakKeyDictionary
from ..core.types import ID
from ..document.document import Document
from ..model import Model, collect_models
from ..settings import settings
from ..themes.theme import Theme
from ..util.dataclasses import dataclass, field
from ..util.serialization import (
@contextmanager
def OutputDocumentFor(objs: Sequence[Model], apply_theme: Theme | type[FromCurdoc] | None=None, always_new: bool=False) -> Iterator[Document]:
    """ Find or create a (possibly temporary) Document to use for serializing
    Bokeh content.

    Typical usage is similar to:

    .. code-block:: python

         with OutputDocumentFor(models):
            (docs_json, [render_item]) = standalone_docs_json_and_render_items(models)

    Inside the context manager, the models will be considered to be part of a single
    Document, with any theme specified, which can thus be serialized as a unit. Where
    possible, OutputDocumentFor attempts to use an existing Document. However, this is
    not possible in three cases:

    * If passed a series of models that have no Document at all, a new Document will
      be created, and all the models will be added as roots. After the context manager
      exits, the new Document will continue to be the models' document.

    * If passed a subset of Document.roots, then OutputDocumentFor temporarily "re-homes"
      the models in a new bare Document that is only available inside the context manager.

    * If passed a list of models that have different documents, then OutputDocumentFor
      temporarily "re-homes" the models in a new bare Document that is only available
      inside the context manager.

    OutputDocumentFor will also perfom document validation before yielding, if
    ``settings.perform_document_validation()`` is True.


        objs (seq[Model]) :
            a sequence of Models that will be serialized, and need a common document

        apply_theme (Theme or FromCurdoc or None, optional):
            Sets the theme for the doc while inside this context manager. (default: None)

            If None, use whatever theme is on the document that is found or created

            If FromCurdoc, use curdoc().theme, restoring any previous theme afterwards

            If a Theme instance, use that theme, restoring any previous theme afterwards

        always_new (bool, optional) :
            Always return a new document, even in cases where it is otherwise possible
            to use an existing document on models.

    Yields:
        Document

    """
    if not isinstance(objs, Sequence) or len(objs) == 0 or (not all((isinstance(x, Model) for x in objs))):
        raise ValueError('OutputDocumentFor expects a non-empty sequence of Models')

    def finish() -> None:
        pass
    docs = {obj.document for obj in objs if obj.document is not None}
    if always_new:

        def finish() -> None:
            _dispose_temp_doc(objs)
        doc = _create_temp_doc(objs)
    elif len(docs) == 0:
        doc = _new_doc()
        for model in objs:
            doc.add_root(model)
    elif len(docs) == 1:
        doc = docs.pop()
        if set(objs) != set(doc.roots):

            def finish() -> None:
                _dispose_temp_doc(objs)
            doc = _create_temp_doc(objs)
        pass
    else:

        def finish():
            _dispose_temp_doc(objs)
        doc = _create_temp_doc(objs)
    if settings.perform_document_validation():
        doc.validate()
    _set_temp_theme(doc, apply_theme)
    yield doc
    _unset_temp_theme(doc)
    finish()