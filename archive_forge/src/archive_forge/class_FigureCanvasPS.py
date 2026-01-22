import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi = 72
    filetypes = {'ps': 'Postscript', 'eps': 'Encapsulated Postscript'}

    def get_default_filetype(self):
        return 'ps'

    def _print_ps(self, fmt, outfile, *, metadata=None, papertype=None, orientation='portrait', bbox_inches_restore=None, **kwargs):
        dpi = self.figure.dpi
        self.figure.dpi = 72
        dsc_comments = {}
        if isinstance(outfile, (str, os.PathLike)):
            filename = pathlib.Path(outfile).name
            dsc_comments['Title'] = filename.encode('ascii', 'replace').decode('ascii')
        dsc_comments['Creator'] = (metadata or {}).get('Creator', f'Matplotlib v{mpl.__version__}, https://matplotlib.org/')
        source_date_epoch = os.getenv('SOURCE_DATE_EPOCH')
        dsc_comments['CreationDate'] = datetime.datetime.fromtimestamp(int(source_date_epoch), datetime.timezone.utc).strftime('%a %b %d %H:%M:%S %Y') if source_date_epoch else time.ctime()
        dsc_comments = '\n'.join((f'%%{k}: {v}' for k, v in dsc_comments.items()))
        if papertype is None:
            papertype = mpl.rcParams['ps.papersize']
        papertype = papertype.lower()
        _api.check_in_list(['figure', 'auto', *papersize], papertype=papertype)
        orientation = _api.check_getitem(_Orientation, orientation=orientation.lower())
        printer = self._print_figure_tex if mpl.rcParams['text.usetex'] else self._print_figure
        printer(fmt, outfile, dpi=dpi, dsc_comments=dsc_comments, orientation=orientation, papertype=papertype, bbox_inches_restore=bbox_inches_restore, **kwargs)

    def _print_figure(self, fmt, outfile, *, dpi, dsc_comments, orientation, papertype, bbox_inches_restore=None):
        """
        Render the figure to a filesystem path or a file-like object.

        Parameters are as for `.print_figure`, except that *dsc_comments* is a
        string containing Document Structuring Convention comments,
        generated from the *metadata* parameter to `.print_figure`.
        """
        is_eps = fmt == 'eps'
        if not (isinstance(outfile, (str, os.PathLike)) or is_writable_file_like(outfile)):
            raise ValueError('outfile must be a path or a file-like object')
        width, height = self.figure.get_size_inches()
        if papertype == 'auto':
            papertype = _get_papertype(*orientation.swap_if_landscape((width, height)))
        if is_eps or papertype == 'figure':
            paper_width, paper_height = (width, height)
        else:
            paper_width, paper_height = orientation.swap_if_landscape(papersize[papertype])
        xo = 72 * 0.5 * (paper_width - width)
        yo = 72 * 0.5 * (paper_height - height)
        llx = xo
        lly = yo
        urx = llx + self.figure.bbox.width
        ury = lly + self.figure.bbox.height
        rotation = 0
        if orientation is _Orientation.landscape:
            llx, lly, urx, ury = (lly, llx, ury, urx)
            xo, yo = (72 * paper_height - yo, xo)
            rotation = 90
        bbox = (llx, lly, urx, ury)
        self._pswriter = StringIO()
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(self.figure, width, height, dpi, ps_renderer, bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)

        def print_figure_impl(fh):
            if is_eps:
                print('%!PS-Adobe-3.0 EPSF-3.0', file=fh)
            else:
                print('%!PS-Adobe-3.0', file=fh)
                if papertype != 'figure':
                    print(f'%%DocumentPaperSizes: {papertype}', file=fh)
                print('%%Pages: 1', file=fh)
            print(f'%%LanguageLevel: 3\n{dsc_comments}\n%%Orientation: {orientation.name}\n{get_bbox_header(bbox)[0]}\n%%EndComments\n', end='', file=fh)
            Ndict = len(_psDefs)
            print('%%BeginProlog', file=fh)
            if not mpl.rcParams['ps.useafm']:
                Ndict += len(ps_renderer._character_tracker.used)
            print('/mpldict %d dict def' % Ndict, file=fh)
            print('mpldict begin', file=fh)
            print('\n'.join(_psDefs), file=fh)
            if not mpl.rcParams['ps.useafm']:
                for font_path, chars in ps_renderer._character_tracker.used.items():
                    if not chars:
                        continue
                    fonttype = mpl.rcParams['ps.fonttype']
                    if len(chars) > 255:
                        fonttype = 42
                    fh.flush()
                    if fonttype == 3:
                        fh.write(_font_to_ps_type3(font_path, chars))
                    else:
                        _font_to_ps_type42(font_path, chars, fh)
            print('end', file=fh)
            print('%%EndProlog', file=fh)
            if not is_eps:
                print('%%Page: 1 1', file=fh)
            print('mpldict begin', file=fh)
            print('%s translate' % _nums_to_str(xo, yo), file=fh)
            if rotation:
                print('%d rotate' % rotation, file=fh)
            print(f'0 0 {_nums_to_str(width * 72, height * 72)} rectclip', file=fh)
            print(self._pswriter.getvalue(), file=fh)
            print('end', file=fh)
            print('showpage', file=fh)
            if not is_eps:
                print('%%EOF', file=fh)
            fh.flush()
        if mpl.rcParams['ps.usedistiller']:
            with TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, 'tmp.ps')
                with open(tmpfile, 'w', encoding='latin-1') as fh:
                    print_figure_impl(fh)
                if mpl.rcParams['ps.usedistiller'] == 'ghostscript':
                    _try_distill(gs_distill, tmpfile, is_eps, ptype=papertype, bbox=bbox)
                elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                    _try_distill(xpdf_distill, tmpfile, is_eps, ptype=papertype, bbox=bbox)
                _move_path_to_path_or_stream(tmpfile, outfile)
        else:
            with cbook.open_file_cm(outfile, 'w', encoding='latin-1') as file:
                if not file_requires_unicode(file):
                    file = codecs.getwriter('latin-1')(file)
                print_figure_impl(file)

    def _print_figure_tex(self, fmt, outfile, *, dpi, dsc_comments, orientation, papertype, bbox_inches_restore=None):
        """
        If :rc:`text.usetex` is True, a temporary pair of tex/eps files
        are created to allow tex to manage the text layout via the PSFrags
        package. These files are processed to yield the final ps or eps file.

        The rest of the behavior is as for `._print_figure`.
        """
        is_eps = fmt == 'eps'
        width, height = self.figure.get_size_inches()
        xo = 0
        yo = 0
        llx = xo
        lly = yo
        urx = llx + self.figure.bbox.width
        ury = lly + self.figure.bbox.height
        bbox = (llx, lly, urx, ury)
        self._pswriter = StringIO()
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(self.figure, width, height, dpi, ps_renderer, bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir, 'tmp.ps')
            tmppath.write_text(f'%!PS-Adobe-3.0 EPSF-3.0\n%%LanguageLevel: 3\n{dsc_comments}\n{get_bbox_header(bbox)[0]}\n%%EndComments\n%%BeginProlog\n/mpldict {len(_psDefs)} dict def\nmpldict begin\n{''.join(_psDefs)}\nend\n%%EndProlog\nmpldict begin\n{_nums_to_str(xo, yo)} translate\n0 0 {_nums_to_str(width * 72, height * 72)} rectclip\n{self._pswriter.getvalue()}\nend\nshowpage\n', encoding='latin-1')
            if orientation is _Orientation.landscape:
                width, height = (height, width)
                bbox = (lly, llx, ury, urx)
            if is_eps or papertype == 'figure':
                paper_width, paper_height = orientation.swap_if_landscape(self.figure.get_size_inches())
            else:
                if papertype == 'auto':
                    papertype = _get_papertype(width, height)
                paper_width, paper_height = papersize[papertype]
            psfrag_rotated = _convert_psfrags(tmppath, ps_renderer.psfrag, paper_width, paper_height, orientation.name)
            if mpl.rcParams['ps.usedistiller'] == 'ghostscript' or mpl.rcParams['text.usetex']:
                _try_distill(gs_distill, tmppath, is_eps, ptype=papertype, bbox=bbox, rotated=psfrag_rotated)
            elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                _try_distill(xpdf_distill, tmppath, is_eps, ptype=papertype, bbox=bbox, rotated=psfrag_rotated)
            _move_path_to_path_or_stream(tmppath, outfile)
    print_ps = functools.partialmethod(_print_ps, 'ps')
    print_eps = functools.partialmethod(_print_ps, 'eps')

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()