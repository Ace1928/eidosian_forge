import warnings
from ctypes import *
from .base import Config, CanvasConfig, Context
from pyglet.canvas.xlib import XlibCanvas
from pyglet.gl import glx
from pyglet.gl import glxext_arb
from pyglet.gl import glx_info
from pyglet.gl import glxext_mesa
from pyglet.gl import lib
from pyglet import gl
class XlibContext(Context):

    def __init__(self, config, share):
        super().__init__(config, share)
        self.x_display = config.canvas.display._display
        self.glx_context = self._create_glx_context(share)
        if not self.glx_context:
            raise gl.ContextException('Could not create GL context')
        self._have_SGI_video_sync = config.glx_info.have_extension('GLX_SGI_video_sync')
        self._have_SGI_swap_control = config.glx_info.have_extension('GLX_SGI_swap_control')
        self._have_EXT_swap_control = config.glx_info.have_extension('GLX_EXT_swap_control')
        self._have_MESA_swap_control = config.glx_info.have_extension('GLX_MESA_swap_control')
        self._use_video_sync = self._have_SGI_video_sync and (not (self._have_EXT_swap_control or self._have_MESA_swap_control))
        self._vsync = True
        self.glx_window = None

    def is_direct(self):
        return glx.glXIsDirect(self.x_display, self.glx_context)

    def _create_glx_context(self, share):
        if share:
            share_context = share.glx_context
        else:
            share_context = None
        attribs = []
        if self.config.major_version is not None:
            attribs.extend([glxext_arb.GLX_CONTEXT_MAJOR_VERSION_ARB, self.config.major_version])
        if self.config.minor_version is not None:
            attribs.extend([glxext_arb.GLX_CONTEXT_MINOR_VERSION_ARB, self.config.minor_version])
        if self.config.opengl_api == 'gl':
            attribs.extend([glxext_arb.GLX_CONTEXT_PROFILE_MASK_ARB, glxext_arb.GLX_CONTEXT_CORE_PROFILE_BIT_ARB])
        elif self.config.opengl_api == 'gles':
            attribs.extend([glxext_arb.GLX_CONTEXT_PROFILE_MASK_ARB, glxext_arb.GLX_CONTEXT_ES2_PROFILE_BIT_EXT])
        flags = 0
        if self.config.forward_compatible:
            flags |= glxext_arb.GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB
        if self.config.debug:
            flags |= glxext_arb.GLX_CONTEXT_DEBUG_BIT_ARB
        if flags:
            attribs.extend([glxext_arb.GLX_CONTEXT_FLAGS_ARB, flags])
        attribs.append(0)
        attribs = (c_int * len(attribs))(*attribs)
        return glxext_arb.glXCreateContextAttribsARB(self.config.canvas.display._display, self.config.fbconfig, share_context, True, attribs)

    def attach(self, canvas):
        if canvas is self.canvas:
            return
        super().attach(canvas)
        self.glx_window = glx.glXCreateWindow(self.x_display, self.config.fbconfig, canvas.x_window, None)
        self.set_current()

    def set_current(self):
        glx.glXMakeContextCurrent(self.x_display, self.glx_window, self.glx_window, self.glx_context)
        super().set_current()

    def detach(self):
        if not self.canvas:
            return
        self.set_current()
        gl.glFlush()
        super().detach()
        glx.glXMakeContextCurrent(self.x_display, 0, 0, None)
        if self.glx_window:
            glx.glXDestroyWindow(self.x_display, self.glx_window)
            self.glx_window = None

    def destroy(self):
        super().destroy()
        if self.glx_window:
            glx.glXDestroyWindow(self.config.display._display, self.glx_window)
            self.glx_window = None
        if self.glx_context:
            glx.glXDestroyContext(self.x_display, self.glx_context)
            self.glx_context = None

    def set_vsync(self, vsync=True):
        self._vsync = vsync
        interval = vsync and 1 or 0
        try:
            if not self._use_video_sync and self._have_EXT_swap_control:
                glxext_arb.glXSwapIntervalEXT(self.x_display, glx.glXGetCurrentDrawable(), interval)
            elif not self._use_video_sync and self._have_MESA_swap_control:
                glxext_mesa.glXSwapIntervalMESA(interval)
            elif self._have_SGI_swap_control:
                glxext_arb.glXSwapIntervalSGI(interval)
        except lib.MissingFunctionException as e:
            warnings.warn(str(e))

    def get_vsync(self):
        return self._vsync

    def _wait_vsync(self):
        if self._vsync and self._have_SGI_video_sync and self._use_video_sync:
            count = c_uint()
            glxext_arb.glXGetVideoSyncSGI(byref(count))
            glxext_arb.glXWaitVideoSyncSGI(2, (count.value + 1) % 2, byref(count))

    def flip(self):
        if not self.glx_window:
            return
        if self._vsync:
            self._wait_vsync()
        glx.glXSwapBuffers(self.x_display, self.glx_window)