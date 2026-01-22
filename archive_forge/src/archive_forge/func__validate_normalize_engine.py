from typing import Literal, Optional, Union, cast, Tuple
from .deprecation import AltairDeprecationWarning
from .html import spec_to_html
from ._importers import import_vl_convert, vl_version_for_vl_convert
import struct
import warnings
def _validate_normalize_engine(engine: Optional[Literal['vl-convert', 'altair_saver']], format: Literal['png', 'svg', 'pdf', 'vega']) -> str:
    """Helper to validate and normalize the user-provided engine

    engine : {None, 'vl-convert', 'altair_saver'}
        the user-provided engine string
    format : string {'png', 'svg', 'pdf', 'vega'}
        the format of the mimebundle to be returned
    """
    try:
        vlc = import_vl_convert()
    except ImportError:
        vlc = None
    try:
        import altair_saver
    except ImportError:
        altair_saver = None
    normalized_engine = None if engine is None else engine.lower().replace('-', '').replace('_', '')
    if normalized_engine == 'vlconvert':
        if vlc is None:
            raise ValueError("The 'vl-convert' conversion engine requires the vl-convert-python package")
    elif normalized_engine == 'altairsaver':
        if altair_saver is None:
            raise ValueError("The 'altair_saver' conversion engine requires the altair_saver package")
    elif normalized_engine is None:
        if vlc is not None:
            normalized_engine = 'vlconvert'
        elif altair_saver is not None:
            normalized_engine = 'altairsaver'
        else:
            raise ValueError('Saving charts in {fmt!r} format requires the vl-convert-python or altair_saver package: see http://github.com/altair-viz/altair_saver/'.format(fmt=format))
    else:
        raise ValueError('Invalid conversion engine {engine!r}. Expected one of {valid!r}'.format(engine=engine, valid=('vl-convert', 'altair_saver')))
    return normalized_engine