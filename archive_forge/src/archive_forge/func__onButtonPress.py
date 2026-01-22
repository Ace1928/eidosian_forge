import gtk
def _onButtonPress(self, widget, event):
    if event.type == gtk.gdk._2BUTTON_PRESS:
        print(['Double click!'])
        self._onReplace()