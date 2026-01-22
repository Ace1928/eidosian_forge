import re
from typing import Dict, Any
def append_paragraph(self):
    last_token = self.last_token()
    if last_token and last_token['type'] == 'paragraph':
        pos = self.find_line_end()
        last_token['text'] += self.get_text(pos)
        return pos