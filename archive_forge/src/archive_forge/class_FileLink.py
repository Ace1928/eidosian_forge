from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
class FileLink(object):
    """Class for embedding a local file link in an IPython session, based on path

    e.g. to embed a link that was generated in the IPython notebook as my/data.txt

    you would do::

        local_file = FileLink("my/data.txt")
        display(local_file)

    or in the HTML notebook, just::

        FileLink("my/data.txt")
    """
    html_link_str = "<a href='%s' target='_blank'>%s</a>"

    def __init__(self, path, url_prefix='', result_html_prefix='', result_html_suffix='<br>'):
        """
        Parameters
        ----------
        path : str
            path to the file or directory that should be formatted
        url_prefix : str
            prefix to be prepended to all files to form a working link [default:
            '']
        result_html_prefix : str
            text to append to beginning to link [default: '']
        result_html_suffix : str
            text to append at the end of link [default: '<br>']
        """
        if isdir(path):
            raise ValueError("Cannot display a directory using FileLink. Use FileLinks to display '%s'." % path)
        self.path = fsdecode(path)
        self.url_prefix = url_prefix
        self.result_html_prefix = result_html_prefix
        self.result_html_suffix = result_html_suffix

    def _format_path(self):
        fp = ''.join([self.url_prefix, html_escape(self.path)])
        return ''.join([self.result_html_prefix, self.html_link_str % (fp, html_escape(self.path, quote=False)), self.result_html_suffix])

    def _repr_html_(self):
        """return html link to file
        """
        if not exists(self.path):
            return "Path (<tt>%s</tt>) doesn't exist. It may still be in the process of being generated, or you may have the incorrect path." % self.path
        return self._format_path()

    def __repr__(self):
        """return absolute path to file
        """
        return abspath(self.path)