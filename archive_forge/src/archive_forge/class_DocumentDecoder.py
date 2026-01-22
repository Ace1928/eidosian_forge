from os.path import dirname as _dirname
from os.path import splitext as _splitext
import pyglet
from pyglet.text import layout, document, caret
class DocumentDecoder:
    """Abstract document decoder.
    """

    def decode(self, text, location=None):
        """Decode document text.
        
        :Parameters:
            `text` : str
                Text to decode
            `location` : `Location`
                Location to use as base path for additional resources
                referenced within the document (for example, HTML images).

        :rtype: `AbstractDocument`
        """
        raise NotImplementedError('abstract')