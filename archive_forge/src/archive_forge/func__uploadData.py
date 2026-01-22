from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from ..GLGraphicsItem import GLGraphicsItem
def _uploadData(self):
    glEnable(GL_TEXTURE_3D)
    if self.texture is None:
        self.texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_3D, self.texture)
    if self.smooth:
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    else:
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
    shape = self.data.shape
    glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_RGBA, shape[0], shape[1], shape[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0, GL_TEXTURE_WIDTH) == 0:
        raise Exception('OpenGL failed to create 3D texture (%dx%dx%d); too large for this hardware.' % shape[:3])
    data = np.ascontiguousarray(self.data.transpose((2, 1, 0, 3)))
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, shape[0], shape[1], shape[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glDisable(GL_TEXTURE_3D)
    self.lists = {}
    for ax in [0, 1, 2]:
        for d in [-1, 1]:
            l = glGenLists(1)
            self.lists[ax, d] = l
            glNewList(l, GL_COMPILE)
            self.drawVolume(ax, d)
            glEndList()
    self._needUpload = False