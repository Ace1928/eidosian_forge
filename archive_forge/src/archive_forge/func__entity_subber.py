import html
import html.entities
import re
from urllib.parse import quote, unquote
def _entity_subber(match, name2c=html.entities.name2codepoint):
    code = name2c.get(match.group(1))
    if code:
        return chr(code)
    else:
        return match.group(0)