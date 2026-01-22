from __future__ import annotations
import functools
import json
import typing as t
from io import BytesIO
from .._internal import _wsgi_decoding_dance
from ..datastructures import CombinedMultiDict
from ..datastructures import EnvironHeaders
from ..datastructures import FileStorage
from ..datastructures import ImmutableMultiDict
from ..datastructures import iter_multi_items
from ..datastructures import MultiDict
from ..exceptions import BadRequest
from ..exceptions import UnsupportedMediaType
from ..formparser import default_stream_factory
from ..formparser import FormDataParser
from ..sansio.request import Request as _SansIORequest
from ..utils import cached_property
from ..utils import environ_property
from ..wsgi import _get_server
from ..wsgi import get_input_stream
def _load_form_data(self) -> None:
    """Method used internally to retrieve submitted data.  After calling
        this sets `form` and `files` on the request object to multi dicts
        filled with the incoming form data.  As a matter of fact the input
        stream will be empty afterwards.  You can also call this method to
        force the parsing of the form data.

        .. versionadded:: 0.8
        """
    if 'form' in self.__dict__:
        return
    if self.want_form_data_parsed:
        parser = self.make_form_data_parser()
        data = parser.parse(self._get_stream_for_parsing(), self.mimetype, self.content_length, self.mimetype_params)
    else:
        data = (self.stream, self.parameter_storage_class(), self.parameter_storage_class())
    d = self.__dict__
    d['stream'], d['form'], d['files'] = data