from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def tkKeyPress(self, event):
    if self.mouse_mode:
        return
    k = event.keysym.lower()
    t = time.time()
    cursor = _cursor_mappings.get(k)
    if cursor:
        self.configure(cursor=cursor)
    last_and_release = self.key_to_last_accounted_and_release_time.get(k)
    if last_and_release:
        if last_and_release[0] is None:
            last_and_release[0] = t
        last_and_release[1] = None
        self.schedule_process_key_events_and_redraw(1)
    if event.keysym == 'u':
        print('View SO(1,3)-matrix and current tetrahedron:', self.view_state)
    if event.keysym == 'v':
        self.view = (self.view + 1) % 3
        print('Color for rays that have not hit geometry:', _viewModes[self.view])
        self.redraw_if_initialized()
    if event.keysym == 'p':
        from snappy.CyOpenGL import get_gl_string
        self.make_current()
        for k in ['GL_VERSION', 'GL_SHADING_LANGUAGE_VERSION']:
            print('%s: %s' % (k, get_gl_string(k)))
    if event.keysym == 'm':
        width = 1000
        height = 1000
        f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        self.save_image(width, height, f)
        print('Image saved to: ', f.name)