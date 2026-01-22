import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
class Opengl(RawOpengl):
    """Tkinter bindings for an Opengl widget.
Mike Hartshorn
Department of Chemistry
University of York, UK
http://www.yorvic.york.ac.uk/~mjh/
"""

    def __init__(self, master=None, cnf={}, **kw):
        """        Create an opengl widget.
        Arrange for redraws when the window is exposed or when
        it changes size."""
        RawOpengl.__init__(*(self, master, cnf), **kw)
        self.initialised = 0
        self.xmouse = 0
        self.ymouse = 0
        self.xcenter = 0.0
        self.ycenter = 0.0
        self.zcenter = 0.0
        self.r_back = 1.0
        self.g_back = 0.0
        self.b_back = 1.0
        self.distance = 10.0
        self.fovy = 30.0
        self.near = 0.1
        self.far = 1000.0
        self.autospin_allowed = 0
        self.autospin = 0
        self.bind('<Map>', self.tkMap)
        self.bind('<Expose>', self.tkExpose)
        self.bind('<Configure>', self.tkExpose)
        self.bind('<Shift-Button-1>', self.tkHandlePick)
        self.bind('<Button-1>', self.tkRecordMouse)
        self.bind('<B1-Motion>', self.tkTranslate)
        self.bind('<Button-2>', self.StartRotate)
        self.bind('<B2-Motion>', self.tkRotate)
        self.bind('<ButtonRelease-2>', self.tkAutoSpin)
        self.bind('<Button-3>', self.tkRecordMouse)
        self.bind('<B3-Motion>', self.tkScale)

    def help(self):
        """Help for the widget."""
        d = dialog.Dialog(None, {'title': 'Viewer help', 'text': 'Button-1: Translate\nButton-2: Rotate\nButton-3: Zoom\nReset: Resets transformation to identity\n', 'bitmap': 'questhead', 'default': 0, 'strings': ('Done', 'Ok')})
        assert d

    def activate(self):
        """Cause this Opengl widget to be the current destination for drawing."""
        self.tk.call(self._w, 'makecurrent')

    def basic_lighting(self):
        """        Set up some basic lighting (single infinite light source).

        Also switch on the depth buffer."""
        self.activate()
        light_position = (1, 1, 1, 0)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_background(self, r, g, b):
        """Change the background colour of the widget."""
        self.r_back = r
        self.g_back = g
        self.b_back = b
        self.tkRedraw()

    def set_centerpoint(self, x, y, z):
        """Set the new center point for the model.
        This is where we are looking."""
        self.xcenter = x
        self.ycenter = y
        self.zcenter = z
        self.tkRedraw()

    def set_eyepoint(self, distance):
        """Set how far the eye is from the position we are looking."""
        self.distance = distance
        self.tkRedraw()

    def reset(self):
        """Reset rotation matrix for this widget."""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.tkRedraw()

    def tkHandlePick(self, event):
        """Handle a pick on the scene."""
        if hasattr(self, 'pick'):
            realy = self.winfo_height() - event.y
            p1 = gluUnProject(event.x, realy, 0.0)
            p2 = gluUnProject(event.x, realy, 1.0)
            if self.pick(self, p1, p2):
                'If the pick method returns true we redraw the scene.'
                self.tkRedraw()

    def tkRecordMouse(self, event):
        """Record the current mouse position."""
        self.xmouse = event.x
        self.ymouse = event.y

    def StartRotate(self, event):
        self.autospin = 0
        self.tkRecordMouse(event)

    def tkScale(self, event):
        """Scale the scene.  Achieved by moving the eye position.

        Dragging up zooms in, while dragging down zooms out
        """
        scale = 1 - 0.01 * (event.y - self.ymouse)
        if scale < 0.001:
            scale = 0.001
        elif scale > 1000:
            scale = 1000
        self.distance = self.distance * scale
        self.tkRedraw()
        self.tkRecordMouse(event)

    def do_AutoSpin(self):
        self.activate()
        glRotateScene(0.5, self.xcenter, self.ycenter, self.zcenter, self.yspin, self.xspin, 0, 0)
        self.tkRedraw()
        if self.autospin:
            self.after(10, self.do_AutoSpin)

    def tkAutoSpin(self, event):
        """Perform autospin of scene."""
        self.after(4)
        self.update_idletasks()
        x = self.tk.getint(self.tk.call('winfo', 'pointerx', self._w))
        y = self.tk.getint(self.tk.call('winfo', 'pointery', self._w))
        if self.autospin_allowed:
            if x != event.x_root and y != event.y_root:
                self.autospin = 1
        self.yspin = x - event.x_root
        self.xspin = y - event.y_root
        self.after(10, self.do_AutoSpin)

    def tkRotate(self, event):
        """Perform rotation of scene."""
        self.activate()
        glRotateScene(0.5, self.xcenter, self.ycenter, self.zcenter, event.x, event.y, self.xmouse, self.ymouse)
        self.tkRedraw()
        self.tkRecordMouse(event)

    def tkTranslate(self, event):
        """Perform translation of scene."""
        self.activate()
        win_height = max(1, self.winfo_height())
        obj_c = (self.xcenter, self.ycenter, self.zcenter)
        win = gluProject(obj_c[0], obj_c[1], obj_c[2])
        obj = gluUnProject(win[0], win[1] + 0.5 * win_height, win[2])
        dist = math.sqrt(v3distsq(obj, obj_c))
        scale = abs(dist / (0.5 * win_height))
        glTranslateScene(scale, event.x, event.y, self.xmouse, self.ymouse)
        self.tkRedraw()
        self.tkRecordMouse(event)

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

    def redraw(self, *args, **named):
        """Prevent access errors if user doesn't set redraw fast enough"""

    def tkMap(self, *dummy):
        """Cause the opengl widget to redraw itself."""
        self.tkExpose()

    def tkExpose(self, *dummy):
        """Redraw the widget.
        Make it active, update tk events, call redraw procedure and
        swap the buffers.  Note: swapbuffers is clever enough to
        only swap double buffered visuals."""
        self.activate()
        if not self.initialised:
            self.basic_lighting()
            self.initialised = 1
        self.tkRedraw()

    def tkPrint(self, file):
        """Turn the current scene into PostScript via the feedback buffer."""
        self.activate()