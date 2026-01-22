import functools
import uuid
class SlugConverter(StringConverter):
    regex = '[-a-zA-Z0-9_]+'