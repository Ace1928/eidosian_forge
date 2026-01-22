from kivy.core.image import Image
from kivy.graphics import Color, Rectangle
from kivy import kivy_data_dir
from os.path import join
def _touch_up(win, touch):
    if touch.grab_current is win:
        ud = touch.ud
        win.canvas.after.remove(ud['tr.color'])
        win.canvas.after.remove(ud['tr.rect'])
        if ud.get('tr.grab') is True:
            touch.ungrab(win)
            ud['tr.grab'] = False