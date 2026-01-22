import numpy
from . import ClusterUtils
def ClusterToImg(cluster, fileName, size=(300, 300), ptColors=[], lineWidth=None, showIndices=0, stopAtCentroids=0, logScale=0):
    """ handles the work of drawing a cluster tree to an image file

    **Arguments**

      - cluster: the cluster tree to be drawn

      - fileName: the name of the file to be created

      - size: the size of output canvas

      - ptColors: if this is specified, the _colors_ will be used to color
        the terminal nodes of the cluster tree.  (color == _pid.Color_)

      - lineWidth: if specified, it will be used for the widths of the lines
        used to draw the tree

   **Notes**

     - The extension on  _fileName_ determines the type of image file created.
       All formats supported by PIL can be used.

     - if _ptColors_ is the wrong length for the number of possible terminal
       node types, this will throw an IndexError

     - terminal node types are determined using their _GetData()_ methods

  """
    try:
        from rdkit.sping.PIL import pidPIL
    except ImportError:
        from rdkit.piddle import piddlePIL
        pidPIL = piddlePIL
    canvas = pidPIL.PILCanvas(size, fileName)
    if lineWidth is None:
        lineWidth = VisOpts.lineWidth
    DrawClusterTree(cluster, canvas, size, ptColors=ptColors, lineWidth=lineWidth, showIndices=showIndices, stopAtCentroids=stopAtCentroids, logScale=logScale)
    if fileName:
        canvas.save()
    return canvas