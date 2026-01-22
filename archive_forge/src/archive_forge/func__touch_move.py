from kivy.core.image import Image
from kivy.graphics import Color, Rectangle
from kivy import kivy_data_dir
from os.path import join
def _touch_move(win, touch):
    ud = touch.ud
    if not ud.get('tr.rect', False):
        _touch_down(win, touch)
    ud['tr.rect'].pos = (touch.x - pointer_image.width / 2.0 * pointer_scale, touch.y - pointer_image.height / 2.0 * pointer_scale)