def _clean_translation(self, value):
    """Concatenate a translation value to one long protein string (PRIVATE)."""
    translation_parts = value.split()
    return ''.join(translation_parts)