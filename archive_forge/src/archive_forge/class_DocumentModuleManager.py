from __future__ import annotations
import logging # isort:skip
import sys
import weakref
from types import ModuleType
from typing import TYPE_CHECKING
class DocumentModuleManager:
    """ Keep track of and clean up after modules created while building Bokeh
    Documents.

    """
    _document: weakref.ReferenceType[Document]
    _modules: list[ModuleType]

    def __init__(self, document: Document):
        """

        Args:
            document (Document): A Document to manage modules for
                A weak reference to the Document will be retained

        """
        self._document = weakref.ref(document)
        self._modules = []

    def __len__(self) -> int:
        return len(self._modules)

    def add(self, module: ModuleType) -> None:
        """ Add a module associated with a Document.

        .. note::
            This method will install the module in ``sys.modules``

        Args:
            module (Module) : a module to install for the configured Document

        Returns:
            None

        """
        if module.__name__ in sys.modules:
            raise RuntimeError(f'Add called already-added module {module.__name__!r} for {self._document()!r}')
        sys.modules[module.__name__] = module
        self._modules.append(module)

    def destroy(self) -> None:
        """ Clean up any added modules, and check that there are no unexpected
        referrers afterwards.

        Returns:
            None

        """
        from gc import get_referrers
        from types import FrameType
        log.debug(f'Deleting {len(self._modules)} modules for document {self._document()!r}')
        for module in self._modules:
            referrers = get_referrers(module)
            referrers = [x for x in referrers if x is not sys.modules]
            referrers = [x for x in referrers if x is not self._modules]
            referrers = [x for x in referrers if not isinstance(x, FrameType)]
            if len(referrers) != 0:
                log.error(f'Module {module!r} has extra unexpected referrers! This could indicate a serious memory leak. Extra referrers: {referrers!r}')
            if module.__name__ in sys.modules:
                del sys.modules[module.__name__]
            module.__dict__.clear()
            del module
        self._modules = []