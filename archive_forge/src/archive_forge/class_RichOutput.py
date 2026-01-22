import sys
from io import StringIO
class RichOutput(object):

    def __init__(self, data=None, metadata=None, transient=None, update=False):
        self.data = data or {}
        self.metadata = metadata or {}
        self.transient = transient or {}
        self.update = update

    def display(self):
        from IPython.display import publish_display_data
        publish_display_data(data=self.data, metadata=self.metadata, transient=self.transient, update=self.update)

    def _repr_mime_(self, mime):
        if mime not in self.data:
            return
        data = self.data[mime]
        if mime in self.metadata:
            return (data, self.metadata[mime])
        else:
            return data

    def _repr_mimebundle_(self, include=None, exclude=None):
        return (self.data, self.metadata)

    def _repr_html_(self):
        return self._repr_mime_('text/html')

    def _repr_latex_(self):
        return self._repr_mime_('text/latex')

    def _repr_json_(self):
        return self._repr_mime_('application/json')

    def _repr_javascript_(self):
        return self._repr_mime_('application/javascript')

    def _repr_png_(self):
        return self._repr_mime_('image/png')

    def _repr_jpeg_(self):
        return self._repr_mime_('image/jpeg')

    def _repr_svg_(self):
        return self._repr_mime_('image/svg+xml')