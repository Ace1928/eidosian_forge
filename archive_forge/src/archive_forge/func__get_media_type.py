def _get_media_type(self):
    """Returns the media 'type/subtype' string, without parameters."""
    return '%s/%s' % (self.major, self.minor)