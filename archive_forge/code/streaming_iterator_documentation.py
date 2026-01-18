from requests.utils import super_len
from .multipart.encoder import CustomBytesIO, encode_with

    This class provides a way of allowing iterators with a known size to be
    streamed instead of chunked.

    In requests, if you pass in an iterator it assumes you want to use
    chunked transfer-encoding to upload the data, which not all servers
    support well. Additionally, you may want to set the content-length
    yourself to avoid this but that will not work. The only way to preempt
    requests using a chunked transfer-encoding and forcing it to stream the
    uploads is to mimic a very specific interace. Instead of having to know
    these details you can instead just use this class. You simply provide the
    size and iterator and pass the instance of StreamingIterator to requests
    via the data parameter like so:

    .. code-block:: python

        from requests_toolbelt import StreamingIterator

        import requests

        # Let iterator be some generator that you already have and size be
        # the size of the data produced by the iterator

        r = requests.post(url, data=StreamingIterator(size, iterator))

    You can also pass file-like objects to :py:class:`StreamingIterator` in
    case requests can't determize the filesize itself. This is the case with
    streaming file objects like ``stdin`` or any sockets. Wrapping e.g. files
    that are on disk with ``StreamingIterator`` is unnecessary, because
    requests can determine the filesize itself.

    Naturally, you should also set the `Content-Type` of your upload
    appropriately because the toolbelt will not attempt to guess that for you.
    