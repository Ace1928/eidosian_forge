from kivy.compat import PY2
from kivy.core.text import LabelBase
def _get_font_id(self):
    return '|'.join([str(self.options[x]) for x in ('font_size', 'font_name_r', 'bold', 'italic', 'underline', 'strikethrough')])