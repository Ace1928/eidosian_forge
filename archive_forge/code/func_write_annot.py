import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def write_annot(filepath, labels, ctab, names, fill_ctab=True):
    """Write out a "new-style" Freesurfer annotation file.

    Note that the color table ``ctab`` is in RGBT form, where T (transparency)
    is 255 - alpha.

    See:
     * https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation
     * https://github.com/freesurfer/freesurfer/blob/dev/matlab/write_annotation.m
     * https://github.com/freesurfer/freesurfer/blob/8b88b34/utils/colortab.c

    Parameters
    ----------
    filepath : str
        Path to annotation file to be written
    labels : ndarray, shape (n_vertices,)
        Annotation id at each vertex.
    ctab : ndarray, shape (n_labels, 5)
        RGBT + label id colortable array.
    names : list of str
        The names of the labels. The length of the list is n_labels.
    fill_ctab : {True, False} optional
        If True, the annotation values for each vertex  are automatically
        generated. In this case, the provided `ctab` may have shape
        (n_labels, 4) or (n_labels, 5) - if the latter, the final column is
        ignored.
    """
    with open(filepath, 'wb') as fobj:
        dt = _ANNOT_DT
        vnum = len(labels)

        def write(num, dtype=dt):
            np.array([num], dtype).tofile(fobj)

        def write_string(s):
            s = (s if isinstance(s, bytes) else s.encode()) + b'\x00'
            write(len(s))
            write(s, dtype='|S%d' % len(s))
        if fill_ctab:
            ctab = np.hstack((ctab[:, :4], _pack_rgb(ctab[:, :3])))
        elif not np.array_equal(ctab[:, [4]], _pack_rgb(ctab[:, :3])):
            warnings.warn(f'Annotation values in {filepath} will be incorrect')
        write(vnum)
        clut_labels = ctab[:, -1][labels]
        clut_labels[np.where(labels == -1)] = 0
        data = np.vstack((np.array(range(vnum)), clut_labels)).T.astype(dt)
        data.tofile(fobj)
        write(1)
        write(-2)
        write(max(np.max(labels) + 1, ctab.shape[0]))
        write_string('NOFILE')
        write(ctab.shape[0])
        for ind, (clu, name) in enumerate(zip(ctab, names)):
            write(ind)
            write_string(name)
            for val in clu[:-1]:
                write(val)