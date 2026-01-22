from kivy.core.image import Image
from kivy.graphics import Color, Rectangle
from kivy import kivy_data_dir
from os.path import join
def _touch_down(win, touch):
    ud = touch.ud
    with win.canvas.after:
        ud['tr.color'] = Color(1, 1, 1, pointer_alpha)
        iw, ih = pointer_image.size
        ud['tr.rect'] = Rectangle(pos=(touch.x - pointer_image.width / 2.0 * pointer_scale, touch.y - pointer_image.height / 2.0 * pointer_scale), size=(iw * pointer_scale, ih * pointer_scale), texture=pointer_image.texture)
    if not ud.get('tr.grab', False):
        ud['tr.grab'] = True
        touch.grab(win)