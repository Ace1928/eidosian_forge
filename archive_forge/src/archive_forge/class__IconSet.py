import os.path
class _IconSet(object):
    """An icon set, ie. the mode and state and the pixmap used for each."""

    def __init__(self, iconset, base_dir):
        """Initialise the icon set from an XML tag."""
        self._fallback = self._file_name(iconset.text, base_dir)
        self._use_fallback = True
        self._roles = {}
        for i in iconset:
            file_name = i.text
            if file_name is not None:
                file_name = self._file_name(file_name, base_dir)
            self._roles[i.tag] = file_name
            self._use_fallback = False
        self.icon = None

    @staticmethod
    def _file_name(fname, base_dir):
        """ Convert a relative filename if we have a base directory. """
        fname = fname.replace('\\', '\\\\')
        if base_dir != '' and fname[0] != ':' and (not os.path.isabs(fname)):
            fname = os.path.join(base_dir, fname)
        return fname

    def set_icon(self, icon, qtgui_module):
        """Save the icon and set its attributes."""
        if self._use_fallback:
            icon.addFile(self._fallback)
        else:
            for role, pixmap in self._roles.items():
                if role.endswith('off'):
                    mode = role[:-3]
                    state = qtgui_module.QIcon.Off
                elif role.endswith('on'):
                    mode = role[:-2]
                    state = qtgui_module.QIcon.On
                else:
                    continue
                mode = getattr(qtgui_module.QIcon, mode.title())
                if pixmap:
                    icon.addPixmap(qtgui_module.QPixmap(pixmap), mode, state)
                else:
                    icon.addPixmap(qtgui_module.QPixmap(), mode, state)
        self.icon = icon

    def __eq__(self, other):
        """Compare two icon sets for equality."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self._use_fallback:
            if other._use_fallback:
                return self._fallback == other._fallback
            return False
        if other._use_fallback:
            return False
        return self._roles == other._roles