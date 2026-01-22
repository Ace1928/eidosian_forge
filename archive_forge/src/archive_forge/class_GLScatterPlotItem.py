from OpenGL.GL import *  # noqa
import numpy as np
from ... import functions as fn
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""

    def __init__(self, parentItem=None, **kwds):
        super().__init__(parentItem=parentItem)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = None
        self.size = 10
        self.color = [1.0, 1.0, 1.0, 0.5]
        self.pxMode = True
        self.setData(**kwds)
        self.shader = None

    def setData(self, **kwds):
        """
        Update the data displayed by this item. All arguments are optional; 
        for example it is allowed to update spot positions while leaving 
        colors unchanged, etc.
        
        ====================  ==================================================
        **Arguments:**
        pos                   (N,3) array of floats specifying point locations.
        color                 (N,4) array of floats (0.0-1.0) specifying
                              spot colors OR a tuple of floats specifying
                              a single color for all spots.
        size                  (N,) array of floats specifying spot sizes or 
                              a single value to apply to all spots.
        pxMode                If True, spot sizes are expressed in pixels. 
                              Otherwise, they are expressed in item coordinates.
        ====================  ==================================================
        """
        args = ['pos', 'color', 'size', 'pxMode']
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        if 'pos' in kwds:
            pos = kwds.pop('pos')
            self.pos = np.ascontiguousarray(pos, dtype=np.float32)
        if 'color' in kwds:
            color = kwds.pop('color')
            if isinstance(color, np.ndarray):
                color = np.ascontiguousarray(color, dtype=np.float32)
            self.color = color
        if 'size' in kwds:
            size = kwds.pop('size')
            if isinstance(size, np.ndarray):
                size = np.ascontiguousarray(size, dtype=np.float32)
            self.size = size
        self.pxMode = kwds.get('pxMode', self.pxMode)
        self.update()

    def initializeGL(self):
        if self.shader is not None:
            return
        w = 64

        def genTexture(x, y):
            r = np.hypot(x - (w - 1) / 2.0, y - (w - 1) / 2.0)
            return 255 * (w / 2 - fn.clip_array(r, w / 2 - 1, w / 2))
        pData = np.empty((w, w, 4))
        pData[:] = 255
        pData[:, :, 3] = np.fromfunction(genTexture, pData.shape[:2])
        pData = pData.astype(np.ubyte)
        if getattr(self, 'pointTexture', None) is None:
            self.pointTexture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pData.shape[0], pData.shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, pData)
        self.shader = shaders.getShaderProgram('pointSprite')

    def paint(self):
        if self.pos is None:
            return
        self.setupGLState()
        glEnable(GL_POINT_SPRITE)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        with self.shader:
            glEnableClientState(GL_VERTEX_ARRAY)
            try:
                pos = self.pos
                glVertexPointerf(pos)
                if isinstance(self.color, np.ndarray):
                    glEnableClientState(GL_COLOR_ARRAY)
                    glColorPointerf(self.color)
                else:
                    color = self.color
                    if isinstance(color, QtGui.QColor):
                        color = color.getRgbF()
                    glColor4f(*color)
                if not self.pxMode or isinstance(self.size, np.ndarray):
                    glEnableClientState(GL_NORMAL_ARRAY)
                    norm = np.zeros(pos.shape, dtype=np.float32)
                    if self.pxMode:
                        norm[..., 0] = self.size
                    else:
                        gpos = self.mapToView(pos.transpose()).transpose()
                        if self.view():
                            pxSize = self.view().pixelSize(gpos)
                        else:
                            pxSize = self.parentItem().view().pixelSize(gpos)
                        norm[..., 0] = self.size / pxSize
                    glNormalPointerf(norm)
                else:
                    glNormal3f(self.size, 0, 0)
                glDrawArrays(GL_POINTS, 0, pos.shape[0])
            finally:
                glDisableClientState(GL_NORMAL_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)
                glDisable(GL_TEXTURE_2D)