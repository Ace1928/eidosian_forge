import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkRedraw(self, *dummy):
    """Cause the opengl widget to redraw itself."""
    if not self.initialised:
        return
    self.activate()
    glPushMatrix()
    self.update_idletasks()
    self.activate()
    w = self.winfo_width()
    h = self.winfo_height()
    glViewport(0, 0, w, h)
    glClearColor(self.r_back, self.g_back, self.b_back, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(self.fovy, float(w) / float(h), self.near, self.far)
    if 0:
        glMatrixMode(GL_MODELVIEW)
        mat = glGetDoublev(GL_MODELVIEW_MATRIX)
        glLoadIdentity()
        glTranslatef(-self.xcenter, -self.ycenter, -(self.zcenter + self.distance))
        glMultMatrixd(mat)
    else:
        gluLookAt(self.xcenter, self.ycenter, self.zcenter + self.distance, self.xcenter, self.ycenter, self.zcenter, 0.0, 1.0, 0.0)
        glMatrixMode(GL_MODELVIEW)
    self.redraw(self)
    glFlush()
    glPopMatrix()
    self.tk.call(self._w, 'swapbuffers')