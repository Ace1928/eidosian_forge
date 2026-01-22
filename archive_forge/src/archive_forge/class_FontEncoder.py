from __future__ import annotations
import json
from typing import Iterable
class FontEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Font):
            return {'__gradio_font__': True, 'name': obj.name, 'class': 'google' if isinstance(obj, GoogleFont) else 'font'}
        return json.JSONEncoder.default(self, obj)