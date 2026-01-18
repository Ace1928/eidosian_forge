import os
import json
from pathlib import Path
import plotly
from plotly.io._utils import validate_coerce_fig_to_dict
def write_image(fig, file, format=None, scale=None, width=None, height=None, validate=True, engine='auto'):
    """
    Convert a figure to a static image and write it to a file or writeable
    object

    Parameters
    ----------
    fig:
        Figure object or dict representing a figure

    file: str or writeable
        A string representing a local file path or a writeable object
        (e.g. a pathlib.Path object or an open file descriptor)

    format: str or None
        The desired image format. One of
          - 'png'
          - 'jpg' or 'jpeg'
          - 'webp'
          - 'svg'
          - 'pdf'
          - 'eps' (Requires the poppler library to be installed and on the PATH)

        If not specified and `file` is a string then this will default to the
        file extension. If not specified and `file` is not a string then this
        will default to:
            - `plotly.io.kaleido.scope.default_format` if engine is "kaleido"
            - `plotly.io.orca.config.default_format` if engine is "orca"

    width: int or None
        The width of the exported image in layout pixels. If the `scale`
        property is 1.0, this will also be the width of the exported image
        in physical pixels.

        If not specified, will default to:
            - `plotly.io.kaleido.scope.default_width` if engine is "kaleido"
            - `plotly.io.orca.config.default_width` if engine is "orca"

    height: int or None
        The height of the exported image in layout pixels. If the `scale`
        property is 1.0, this will also be the height of the exported image
        in physical pixels.

        If not specified, will default to:
            - `plotly.io.kaleido.scope.default_height` if engine is "kaleido"
            - `plotly.io.orca.config.default_height` if engine is "orca"

    scale: int or float or None
        The scale factor to use when exporting the figure. A scale factor
        larger than 1.0 will increase the image resolution with respect
        to the figure's layout pixel dimensions. Whereas as scale factor of
        less than 1.0 will decrease the image resolution.

        If not specified, will default to:
            - `plotly.io.kaleido.scope.default_scale` if engine is "kaleido"
            - `plotly.io.orca.config.default_scale` if engine is "orca"

    validate: bool
        True if the figure should be validated before being converted to
        an image, False otherwise.

    engine: str
        Image export engine to use:
         - "kaleido": Use Kaleido for image export
         - "orca": Use Orca for image export
         - "auto" (default): Use Kaleido if installed, otherwise use orca

    Returns
    -------
    None
    """
    if isinstance(file, str):
        path = Path(file)
    elif isinstance(file, Path):
        path = file
    else:
        path = None
    if path is not None and format is None:
        ext = path.suffix
        if ext:
            format = ext.lstrip('.')
        else:
            raise ValueError("\nCannot infer image type from output path '{file}'.\nPlease add a file extension or specify the type using the format parameter.\nFor example:\n\n    >>> import plotly.io as pio\n    >>> pio.write_image(fig, file_path, format='png')\n".format(file=file))
    img_data = to_image(fig, format=format, scale=scale, width=width, height=height, validate=validate, engine=engine)
    if path is None:
        try:
            file.write(img_data)
            return
        except AttributeError:
            pass
        raise ValueError("\nThe 'file' argument '{file}' is not a string, pathlib.Path object, or file descriptor.\n".format(file=file))
    else:
        path.write_bytes(img_data)