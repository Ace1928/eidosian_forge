import contextlib
import re
import xml.dom.minidom as xml
import numpy as np
from .. import debug
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtSvg, QtWidgets
from .Exporter import Exporter
def _generateItemSvg(item, nodes=None, root=None, options=None):
    """This function is intended to work around some issues with Qt's SVG generator
    and SVG in general.

    .. warning::
        This function, while documented, is not considered part of the public
        API. The reason for its documentation is for ease of referencing by
        :func:`~pyqtgraph.GraphicsItem.generateSvg`. There should be no need
        to call this function explicitly.

    1. Qt SVG does not implement clipping paths. This is absurd.
    The solution is to let Qt generate SVG for each item independently,
    then glue them together manually with clipping.  The format Qt generates 
    for all items looks like this:
        
    .. code-block:: xml
    
        <g>
            <g transform="matrix(...)">
                one or more of: <path/> or <polyline/> or <text/>
            </g>
            <g transform="matrix(...)">
                one or more of: <path/> or <polyline/> or <text/>
            </g>
            . . .
        </g>
        
    2. There seems to be wide disagreement over whether path strokes
    should be scaled anisotropically.  Given that both inkscape and 
    illustrator seem to prefer isotropic scaling, we will optimize for
    those cases.

    .. note::
        
        see: http://web.mit.edu/jonas/www/anisotropy/

    3. Qt generates paths using non-scaling-stroke from SVG 1.2, but
    inkscape only supports 1.1.

    Both 2 and 3 can be addressed by drawing all items in world coordinates.

    Parameters
    ----------
    item : :class:`~pyqtgraph.GraphicsItem`
        GraphicsItem to generate SVG of
    nodes : dict of str, optional
        dictionary keyed on graphics item names, values contains the 
        XML elements, by default None
    root : :class:`~pyqtgraph.GraphicsItem`, optional
        root GraphicsItem, if none, assigns to `item`, by default None
    options : dict of str, optional
        Options to be applied to the generated XML, by default None

    Returns
    -------
    tuple
        tuple where first element is XML element, second element is 
        a list of child GraphicItems XML elements
    """
    profiler = debug.Profiler()
    if options is None:
        options = {}
    if nodes is None:
        nodes = {}
    if root is None:
        root = item
    if hasattr(item, 'isVisible') and (not item.isVisible()):
        return None
    with contextlib.suppress(NotImplementedError, AttributeError):
        return item.generateSvg(nodes)
    if isinstance(item, QtWidgets.QGraphicsScene):
        xmlStr = '<g>\n</g>\n'
        doc = xml.parseString(xmlStr)
        childs = [i for i in item.items() if i.parentItem() is None]
    elif item.__class__.paint == QtWidgets.QGraphicsItem.paint:
        xmlStr = '<g>\n</g>\n'
        doc = xml.parseString(xmlStr)
        childs = item.childItems()
    else:
        childs = item.childItems()
        tr = itemTransform(item, item.scene())
        if isinstance(root, QtWidgets.QGraphicsScene):
            rootPos = QtCore.QPoint(0, 0)
        else:
            rootPos = root.scenePos()
        if hasattr(root, 'boundingRect'):
            resize_x = options['width'] / root.boundingRect().width()
            resize_y = options['height'] / root.boundingRect().height()
        else:
            resize_x = resize_y = 1
        tr2 = QtGui.QTransform(resize_x, 0, 0, resize_y, -rootPos.x(), -rootPos.y())
        tr = tr * tr2
        arr = QtCore.QByteArray()
        buf = QtCore.QBuffer(arr)
        svg = QtSvg.QSvgGenerator()
        svg.setOutputDevice(buf)
        dpi = QtGui.QGuiApplication.primaryScreen().logicalDotsPerInchX()
        svg.setResolution(int(dpi))
        p = QtGui.QPainter()
        p.begin(svg)
        if hasattr(item, 'setExportMode'):
            item.setExportMode(True, {'painter': p})
        try:
            p.setTransform(tr)
            opt = QtWidgets.QStyleOptionGraphicsItem()
            if item.flags() & QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption:
                opt.exposedRect = item.boundingRect()
            item.paint(p, opt, None)
        finally:
            p.end()
        doc = xml.parseString(arr.data())
    try:
        g1 = doc.getElementsByTagName('g')[0]
        defs = doc.getElementsByTagName('defs')
        if len(defs) > 0:
            defs = [n for n in defs[0].childNodes if isinstance(n, xml.Element)]
    except:
        print(doc.toxml())
        raise
    profiler('render')
    correctCoordinates(g1, defs, item, options)
    profiler('correct')
    baseName = item.__class__.__name__
    i = 1
    while True:
        name = baseName + '_%d' % i
        if name not in nodes:
            break
        i += 1
    nodes[name] = g1
    g1.setAttribute('id', name)
    childGroup = g1
    if not isinstance(item, QtWidgets.QGraphicsScene) and item.flags() & item.GraphicsItemFlag.ItemClipsChildrenToShape:
        path = QtWidgets.QGraphicsPathItem(item.mapToScene(item.shape()))
        item.scene().addItem(path)
        try:
            pathNode = _generateItemSvg(path, root=root, options=options)[0].getElementsByTagName('path')[0]
        finally:
            item.scene().removeItem(path)
        clip = f'{name}_clip'
        clipNode = g1.ownerDocument.createElement('clipPath')
        clipNode.setAttribute('id', clip)
        clipNode.appendChild(pathNode)
        g1.appendChild(clipNode)
        childGroup = g1.ownerDocument.createElement('g')
        childGroup.setAttribute('clip-path', f'url(#{clip})')
        g1.appendChild(childGroup)
    profiler('clipping')
    childs.sort(key=lambda c: c.zValue())
    for ch in childs:
        csvg = _generateItemSvg(ch, nodes, root, options=options)
        if csvg is None:
            continue
        cg, cdefs = csvg
        childGroup.appendChild(cg)
        defs.extend(cdefs)
    profiler('children')
    return (g1, defs)