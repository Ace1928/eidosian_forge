from __future__ import annotations
import logging # isort:skip
from json import JSONEncoder
from typing import Any
from ..settings import settings
from .serialization import Buffer, Serialized

    Convert an object or a serialized representation to a JSON string.

    This function accepts Python-serializable objects and converts them to
    a JSON string. This function does not perform any advaced serialization,
    in particular it won't serialize Bokeh models or numpy arrays. For that,
    use :class:`bokeh.core.serialization.Serializer` class, which handles
    serialization of all types of objects that may be encountered in Bokeh.

    Args:
        obj (obj) : the object to serialize to JSON format

        pretty (bool, optional) :

            Whether to generate prettified output. If ``True``, spaces are
            added after added after separators, and indentation and newlines
            are applied. (default: False)

            Pretty output can also be enabled with the environment variable
            ``BOKEH_PRETTY``, which overrides this argument, if set.

        indent (int or None, optional) :

            Amount of indentation to use in generated JSON output. If ``None``
            then no indentation is used, unless pretty output is enabled,
            in which case two spaces are used. (default: None)

    Returns:

        str: RFC-8259 JSON string

    Examples:

        .. code-block:: python

            >>> import numpy as np

            >>> from bokeh.core.serialization import Serializer
            >>> from bokeh.core.json_encoder import serialize_json

            >>> s = Serializer()

            >>> obj = dict(b=np.datetime64("2023-02-25"), a=np.arange(3))
            >>> rep = s.encode(obj)
            >>> rep
            {
                'type': 'map',
                'entries': [
                    ('b', 1677283200000.0),
                    ('a', {
                        'type': 'ndarray',
                        'array': {'type': 'bytes', 'data': Buffer(id='p1000', data=<memory at 0x7fe5300e2d40>)},
                        'shape': [3],
                        'dtype': 'int32',
                        'order': 'little',
                    }),
                ],
            }

            >>> serialize_json(rep)
            '{"type":"map","entries":[["b",1677283200000.0],["a",{"type":"ndarray","array":'
            "{"type":"bytes","data":"AAAAAAEAAAACAAAA"},"shape":[3],"dtype":"int32","order":"little"}]]}'

    .. note::

        Using this function isn't strictly necessary. The serializer can be
        configured to produce output that's fully compatible with ``dumps()``
        from the standard library module ``json``. The main difference between
        this function and ``dumps()`` is handling of memory buffers. Use the
        following setup:

        .. code-block:: python

            >>> s = Serializer(deferred=False)

            >>> import json
            >>> json.dumps(s.encode(obj))

    