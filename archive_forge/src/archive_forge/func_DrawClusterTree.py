import numpy
from . import ClusterUtils
def DrawClusterTree(cluster, canvas, size, ptColors=[], lineWidth=None, showIndices=0, showNodes=1, stopAtCentroids=0, logScale=0, tooClose=-1):
    """ handles the work of drawing a cluster tree on a Sping canvas

    **Arguments**

      - cluster: the cluster tree to be drawn

      - canvas:  the Sping canvas on which to draw

      - size: the size of _canvas_

      - ptColors: if this is specified, the _colors_ will be used to color
        the terminal nodes of the cluster tree.  (color == _pid.Color_)

      - lineWidth: if specified, it will be used for the widths of the lines
        used to draw the tree

   **Notes**

     - _Canvas_ is neither _save_d nor _flush_ed at the end of this

     - if _ptColors_ is the wrong length for the number of possible terminal
       node types, this will throw an IndexError

     - terminal node types are determined using their _GetData()_ methods

  """
    renderer = ClusterRenderer(canvas, size, ptColors, lineWidth, showIndices, showNodes, stopAtCentroids, logScale, tooClose)
    renderer.DrawTree(cluster)