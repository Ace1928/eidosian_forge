from __future__ import annotations
from inspect import iscoroutine
from io import BytesIO
from sys import exc_info
from traceback import extract_tb
from types import GeneratorType
from typing import (
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.web._stan import CDATA, CharRef, Comment, Tag, slot, voidElements
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest
def _flattenElement(request: Optional[IRequest], root: Flattenable, write: Callable[[bytes], object], slotData: List[Optional[Mapping[str, Flattenable]]], renderFactory: Optional[IRenderable], dataEscaper: Callable[[Union[bytes, str]], bytes]) -> Generator[Union[Generator[Any, Any, Any], Deferred[Flattenable]], None, None]:
    """
    Make C{root} slightly more flat by yielding all its immediate contents as
    strings, deferreds or generators that are recursive calls to itself.

    @param request: A request object which will be passed to
        L{IRenderable.render}.

    @param root: An object to be made flatter.  This may be of type C{unicode},
        L{str}, L{slot}, L{Tag <twisted.web.template.Tag>}, L{tuple}, L{list},
        L{types.GeneratorType}, L{Deferred}, or an object that implements
        L{IRenderable}.

    @param write: A callable which will be invoked with each L{bytes} produced
        by flattening C{root}.

    @param slotData: A L{list} of L{dict} mapping L{str} slot names to data
        with which those slots will be replaced.

    @param renderFactory: If not L{None}, an object that provides
        L{IRenderable}.

    @param dataEscaper: A 1-argument callable which takes L{bytes} or
        L{unicode} and returns L{bytes}, quoted as appropriate for the
        rendering context.  This is really only one of two values:
        L{attributeEscapingDoneOutside} or L{escapeForContent}, depending on
        whether the rendering context is within an attribute or not.  See the
        explanation in L{writeWithAttributeEscaping}.

    @return: An iterator that eventually writes L{bytes} to C{write}.
        It can yield other iterators or L{Deferred}s; if it yields another
        iterator, the caller will iterate it; if it yields a L{Deferred},
        the result of that L{Deferred} will be another generator, in which
        case it is iterated.  See L{_flattenTree} for the trampoline that
        consumes said values.
    """

    def keepGoing(newRoot: Flattenable, dataEscaper: Callable[[Union[bytes, str]], bytes]=dataEscaper, renderFactory: Optional[IRenderable]=renderFactory, write: Callable[[bytes], object]=write) -> Generator[Union[Flattenable, Deferred[Flattenable]], None, None]:
        return _flattenElement(request, newRoot, write, slotData, renderFactory, dataEscaper)

    def keepGoingAsync(result: Deferred[Flattenable]) -> Deferred[Flattenable]:
        return result.addCallback(keepGoing)
    if isinstance(root, (bytes, str)):
        write(dataEscaper(root))
    elif isinstance(root, slot):
        slotValue = _getSlotValue(root.name, slotData, root.default)
        yield keepGoing(slotValue)
    elif isinstance(root, CDATA):
        write(b'<![CDATA[')
        write(escapedCDATA(root.data))
        write(b']]>')
    elif isinstance(root, Comment):
        write(b'<!--')
        write(escapedComment(root.data))
        write(b'-->')
    elif isinstance(root, Tag):
        slotData.append(root.slotData)
        rendererName = root.render
        if rendererName is not None:
            if renderFactory is None:
                raise ValueError(f'Tag wants to be rendered by method "{rendererName}" but is not contained in any IRenderable')
            rootClone = root.clone(False)
            rootClone.render = None
            renderMethod = renderFactory.lookupRenderMethod(rendererName)
            result = renderMethod(request, rootClone)
            yield keepGoing(result)
            slotData.pop()
            return
        if not root.tagName:
            yield keepGoing(root.children)
            return
        write(b'<')
        if isinstance(root.tagName, str):
            tagName = root.tagName.encode('ascii')
        else:
            tagName = root.tagName
        write(tagName)
        for k, v in root.attributes.items():
            if isinstance(k, str):
                k = k.encode('ascii')
            write(b' ' + k + b'="')
            yield keepGoing(v, attributeEscapingDoneOutside, write=writeWithAttributeEscaping(write))
            write(b'"')
        if root.children or nativeString(tagName) not in voidElements:
            write(b'>')
            yield keepGoing(root.children, escapeForContent)
            write(b'</' + tagName + b'>')
        else:
            write(b' />')
    elif isinstance(root, (tuple, list, GeneratorType)):
        for element in root:
            yield keepGoing(element)
    elif isinstance(root, CharRef):
        escaped = '&#%d;' % (root.ordinal,)
        write(escaped.encode('ascii'))
    elif isinstance(root, Deferred):
        yield keepGoingAsync(_fork(root))
    elif iscoroutine(root):
        yield keepGoingAsync(Deferred.fromCoroutine(cast(Coroutine[Deferred[Flattenable], object, Flattenable], root)))
    elif IRenderable.providedBy(root):
        result = root.render(request)
        yield keepGoing(result, renderFactory=root)
    else:
        raise UnsupportedType(root)