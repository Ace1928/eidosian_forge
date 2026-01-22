from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
def _parse_doc(doc, style=['--']):
    """Parses a help doc for inputs

    Parameters
    ----------
    doc : string
        Documentation string
    style : string default ['--']
        The help command style (--, -)

    Returns
    -------
    optmap : dict of input parameters
    """
    doclist = doc.split('\n')
    optmap = {}
    if isinstance(style, (str, bytes)):
        style = [style]
    for line in doclist:
        linelist = line.split()
        flag = [item for i, item in enumerate(linelist) if i < 2 and any([item.startswith(s) for s in style]) and (len(item) > 1)]
        if flag:
            if len(flag) == 1:
                style_idx = [flag[0].startswith(s) for s in style].index(True)
                flag = flag[0]
            else:
                style_idx = []
                for f in flag:
                    for i, s in enumerate(style):
                        if f.startswith(s):
                            style_idx.append(i)
                            break
                flag = flag[style_idx.index(min(style_idx))]
                style_idx = min(style_idx)
            optmap[flag.split(style[style_idx])[1]] = '%s %%s' % flag
    return optmap