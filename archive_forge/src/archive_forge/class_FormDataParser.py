from __future__ import annotations
import typing as t
from io import BytesIO
from urllib.parse import parse_qsl
from ._internal import _plain_int
from .datastructures import FileStorage
from .datastructures import Headers
from .datastructures import MultiDict
from .exceptions import RequestEntityTooLarge
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartDecoder
from .sansio.multipart import NeedData
from .wsgi import get_content_length
from .wsgi import get_input_stream
class FormDataParser:
    """This class implements parsing of form data for Werkzeug.  By itself
    it can parse multipart and url encoded form data.  It can be subclassed
    and extended but for most mimetypes it is a better idea to use the
    untouched stream and expose it as separate attributes on a request
    object.

    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`Response._get_file_stream`.
    :param max_form_memory_size: the maximum number of bytes to be accepted for
                           in-memory stored form data.  If the data
                           exceeds the value specified an
                           :exc:`~exceptions.RequestEntityTooLarge`
                           exception is raised.
    :param max_content_length: If this is provided and the transmitted data
                               is longer than this value an
                               :exc:`~exceptions.RequestEntityTooLarge`
                               exception is raised.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.
    :param max_form_parts: The maximum number of multipart parts to be parsed. If this
        is exceeded, a :exc:`~exceptions.RequestEntityTooLarge` exception is raised.

    .. versionchanged:: 3.0
        The ``charset`` and ``errors`` parameters were removed.

    .. versionchanged:: 3.0
        The ``parse_functions`` attribute and ``get_parse_func`` methods were removed.

    .. versionchanged:: 2.2.3
        Added the ``max_form_parts`` parameter.

    .. versionadded:: 0.8
    """

    def __init__(self, stream_factory: TStreamFactory | None=None, max_form_memory_size: int | None=None, max_content_length: int | None=None, cls: type[MultiDict] | None=None, silent: bool=True, *, max_form_parts: int | None=None) -> None:
        if stream_factory is None:
            stream_factory = default_stream_factory
        self.stream_factory = stream_factory
        self.max_form_memory_size = max_form_memory_size
        self.max_content_length = max_content_length
        self.max_form_parts = max_form_parts
        if cls is None:
            cls = MultiDict
        self.cls = cls
        self.silent = silent

    def parse_from_environ(self, environ: WSGIEnvironment) -> t_parse_result:
        """Parses the information from the environment as form data.

        :param environ: the WSGI environment to be used for parsing.
        :return: A tuple in the form ``(stream, form, files)``.
        """
        stream = get_input_stream(environ, max_content_length=self.max_content_length)
        content_length = get_content_length(environ)
        mimetype, options = parse_options_header(environ.get('CONTENT_TYPE'))
        return self.parse(stream, content_length=content_length, mimetype=mimetype, options=options)

    def parse(self, stream: t.IO[bytes], mimetype: str, content_length: int | None, options: dict[str, str] | None=None) -> t_parse_result:
        """Parses the information from the given stream, mimetype,
        content length and mimetype parameters.

        :param stream: an input stream
        :param mimetype: the mimetype of the data
        :param content_length: the content length of the incoming data
        :param options: optional mimetype parameters (used for
                        the multipart boundary for instance)
        :return: A tuple in the form ``(stream, form, files)``.

        .. versionchanged:: 3.0
            The invalid ``application/x-url-encoded`` content type is not
            treated as ``application/x-www-form-urlencoded``.
        """
        if mimetype == 'multipart/form-data':
            parse_func = self._parse_multipart
        elif mimetype == 'application/x-www-form-urlencoded':
            parse_func = self._parse_urlencoded
        else:
            return (stream, self.cls(), self.cls())
        if options is None:
            options = {}
        try:
            return parse_func(stream, mimetype, content_length, options)
        except ValueError:
            if not self.silent:
                raise
        return (stream, self.cls(), self.cls())

    def _parse_multipart(self, stream: t.IO[bytes], mimetype: str, content_length: int | None, options: dict[str, str]) -> t_parse_result:
        parser = MultiPartParser(stream_factory=self.stream_factory, max_form_memory_size=self.max_form_memory_size, max_form_parts=self.max_form_parts, cls=self.cls)
        boundary = options.get('boundary', '').encode('ascii')
        if not boundary:
            raise ValueError('Missing boundary')
        form, files = parser.parse(stream, boundary, content_length)
        return (stream, form, files)

    def _parse_urlencoded(self, stream: t.IO[bytes], mimetype: str, content_length: int | None, options: dict[str, str]) -> t_parse_result:
        if self.max_form_memory_size is not None and content_length is not None and (content_length > self.max_form_memory_size):
            raise RequestEntityTooLarge()
        try:
            items = parse_qsl(stream.read().decode(), keep_blank_values=True, errors='werkzeug.url_quote')
        except ValueError as e:
            raise RequestEntityTooLarge() from e
        return (stream, self.cls(items), self.cls())