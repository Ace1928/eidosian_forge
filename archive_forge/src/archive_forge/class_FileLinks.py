from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
class FileLinks(FileLink):
    """Class for embedding local file links in an IPython session, based on path

    e.g. to embed links to files that were generated in the IPython notebook
    under ``my/data``, you would do::

        local_files = FileLinks("my/data")
        display(local_files)

    or in the HTML notebook, just::

        FileLinks("my/data")
    """

    def __init__(self, path, url_prefix='', included_suffixes=None, result_html_prefix='', result_html_suffix='<br>', notebook_display_formatter=None, terminal_display_formatter=None, recursive=True):
        """
        See :class:`FileLink` for the ``path``, ``url_prefix``,
        ``result_html_prefix`` and ``result_html_suffix`` parameters.

        included_suffixes : list
          Filename suffixes to include when formatting output [default: include
          all files]

        notebook_display_formatter : function
          Used to format links for display in the notebook. See discussion of
          formatter functions below.

        terminal_display_formatter : function
          Used to format links for display in the terminal. See discussion of
          formatter functions below.

        Formatter functions must be of the form::

            f(dirname, fnames, included_suffixes)

        dirname : str
          The name of a directory
        fnames : list
          The files in that directory
        included_suffixes : list
          The file suffixes that should be included in the output (passing None
          meansto include all suffixes in the output in the built-in formatters)
        recursive : boolean
          Whether to recurse into subdirectories. Default is True.

        The function should return a list of lines that will be printed in the
        notebook (if passing notebook_display_formatter) or the terminal (if
        passing terminal_display_formatter). This function is iterated over for
        each directory in self.path. Default formatters are in place, can be
        passed here to support alternative formatting.

        """
        if isfile(path):
            raise ValueError("Cannot display a file using FileLinks. Use FileLink to display '%s'." % path)
        self.included_suffixes = included_suffixes
        path = path.rstrip('/')
        self.path = path
        self.url_prefix = url_prefix
        self.result_html_prefix = result_html_prefix
        self.result_html_suffix = result_html_suffix
        self.notebook_display_formatter = notebook_display_formatter or self._get_notebook_display_formatter()
        self.terminal_display_formatter = terminal_display_formatter or self._get_terminal_display_formatter()
        self.recursive = recursive

    def _get_display_formatter(self, dirname_output_format, fname_output_format, fp_format, fp_cleaner=None):
        """generate built-in formatter function

        this is used to define both the notebook and terminal built-in
         formatters as they only differ by some wrapper text for each entry

        dirname_output_format: string to use for formatting directory
         names, dirname will be substituted for a single "%s" which
         must appear in this string
        fname_output_format: string to use for formatting file names,
         if a single "%s" appears in the string, fname will be substituted
         if two "%s" appear in the string, the path to fname will be
          substituted for the first and fname will be substituted for the
          second
        fp_format: string to use for formatting filepaths, must contain
         exactly two "%s" and the dirname will be substituted for the first
         and fname will be substituted for the second
        """

        def f(dirname, fnames, included_suffixes=None):
            result = []
            display_fnames = []
            for fname in fnames:
                if isfile(join(dirname, fname)) and (included_suffixes is None or splitext(fname)[1] in included_suffixes):
                    display_fnames.append(fname)
            if len(display_fnames) == 0:
                pass
            else:
                dirname_output_line = dirname_output_format % dirname
                result.append(dirname_output_line)
                for fname in display_fnames:
                    fp = fp_format % (dirname, fname)
                    if fp_cleaner is not None:
                        fp = fp_cleaner(fp)
                    try:
                        fname_output_line = fname_output_format % (fp, fname)
                    except TypeError:
                        fname_output_line = fname_output_format % fname
                    result.append(fname_output_line)
            return result
        return f

    def _get_notebook_display_formatter(self, spacer='&nbsp;&nbsp;'):
        """ generate function to use for notebook formatting
        """
        dirname_output_format = self.result_html_prefix + '%s/' + self.result_html_suffix
        fname_output_format = self.result_html_prefix + spacer + self.html_link_str + self.result_html_suffix
        fp_format = self.url_prefix + '%s/%s'
        if sep == '\\':

            def fp_cleaner(fp):
                return fp.replace('\\', '/')
        else:
            fp_cleaner = None
        return self._get_display_formatter(dirname_output_format, fname_output_format, fp_format, fp_cleaner)

    def _get_terminal_display_formatter(self, spacer='  '):
        """ generate function to use for terminal formatting
        """
        dirname_output_format = '%s/'
        fname_output_format = spacer + '%s'
        fp_format = '%s/%s'
        return self._get_display_formatter(dirname_output_format, fname_output_format, fp_format)

    def _format_path(self):
        result_lines = []
        if self.recursive:
            walked_dir = list(walk(self.path))
        else:
            walked_dir = [next(walk(self.path))]
        walked_dir.sort()
        for dirname, subdirs, fnames in walked_dir:
            result_lines += self.notebook_display_formatter(dirname, fnames, self.included_suffixes)
        return '\n'.join(result_lines)

    def __repr__(self):
        """return newline-separated absolute paths
        """
        result_lines = []
        if self.recursive:
            walked_dir = list(walk(self.path))
        else:
            walked_dir = [next(walk(self.path))]
        walked_dir.sort()
        for dirname, subdirs, fnames in walked_dir:
            result_lines += self.terminal_display_formatter(dirname, fnames, self.included_suffixes)
        return '\n'.join(result_lines)