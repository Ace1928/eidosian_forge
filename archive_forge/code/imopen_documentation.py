from pathlib import Path
import warnings
from ..config import known_plugins
from ..config.extensions import known_extensions
from .request import (
Open an ImageResource.

    .. warning::
        This warning is for pypy users. If you are not using a context manager,
        remember to deconstruct the returned plugin to avoid leaking the file
        handle to an unclosed file.

    Parameters
    ----------
    uri : str or pathlib.Path or bytes or file or Request
        The :doc:`ImageResource <../../user_guide/requests>` to load the
        image from.
    io_mode : str
        The mode in which the file is opened. Possible values are::

            ``r`` - open the file for reading
            ``w`` - open the file for writing

        Depreciated since v2.9:
        A second character can be added to give the reader a hint on what
        the user expects. This will be ignored by new plugins and will
        only have an effect on legacy plugins. Possible values are::

            ``i`` for a single image,
            ``I`` for multiple images,
            ``v`` for a single volume,
            ``V`` for multiple volumes,
            ``?`` for don't care

    plugin : str, Plugin, or None
        The plugin to use. If set to None imopen will perform a
        search for a matching plugin. If not None, this takes priority over
        the provided format hint.
    extension : str
        If not None, treat the provided ImageResource as if it had the given
        extension. This affects the order in which backends are considered, and
        when writing this may also influence the format used when encoding.
    format_hint : str
        Deprecated. Use `extension` instead.
    legacy_mode : bool
        If true use the v2 behavior when searching for a suitable
        plugin. This will ignore v3 plugins and will check ``plugin``
        against known extensions if no plugin with the given name can be found.
    **kwargs : Any
        Additional keyword arguments will be passed to the plugin upon
        construction.

    Notes
    -----
    Registered plugins are controlled via the ``known_plugins`` dict in
    ``imageio.config``.

    Passing a ``Request`` as the uri is only supported if ``legacy_mode``
    is ``True``. In this case ``io_mode`` is ignored.

    Using the kwarg ``format_hint`` does not enforce the given format. It merely
    provides a `hint` to the selection process and plugin. The selection
    processes uses this hint for optimization; however, a plugin's decision how
    to read a ImageResource will - typically - still be based on the content of
    the resource.


    Examples
    --------

    >>> import imageio.v3 as iio
    >>> with iio.imopen("/path/to/image.png", "r") as file:
    >>>     im = file.read()

    >>> with iio.imopen("/path/to/output.jpg", "w") as file:
    >>>     file.write(im)

    