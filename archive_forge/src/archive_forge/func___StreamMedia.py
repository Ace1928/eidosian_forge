from __future__ import print_function
import email.generator as email_generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import io
import json
import mimetypes
import os
import threading
import six
from six.moves import http_client
from apitools.base.py import buffered_stream
from apitools.base.py import compression
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import stream_slice
from apitools.base.py import util
def __StreamMedia(self, callback=None, finish_callback=None, additional_headers=None, use_chunks=True):
    """Helper function for StreamMedia / StreamInChunks."""
    if self.strategy != RESUMABLE_UPLOAD:
        raise exceptions.InvalidUserInputError('Cannot stream non-resumable upload')
    callback = callback or self.progress_callback
    finish_callback = finish_callback or self.finish_callback
    response = self.__final_response

    def CallSendChunk(start):
        return self.__SendChunk(start, additional_headers=additional_headers)

    def CallSendMediaBody(start):
        return self.__SendMediaBody(start, additional_headers=additional_headers)
    send_func = CallSendChunk if use_chunks else CallSendMediaBody
    if not use_chunks and self.__gzip_encoded:
        raise exceptions.InvalidUserInputError('Cannot gzip encode non-chunked upload')
    if use_chunks:
        self.__ValidateChunksize(self.chunksize)
    self.EnsureInitialized()
    while not self.complete:
        response = send_func(self.stream.tell())
        if response.status_code in (http_client.OK, http_client.CREATED):
            self.__complete = True
            break
        if response.status_code not in (http_client.OK, http_client.CREATED, http_wrapper.RESUME_INCOMPLETE):
            if self.strategy != RESUMABLE_UPLOAD or not self.__IsRetryable(response):
                raise exceptions.HttpError.FromResponse(response)
            self.RefreshResumableUploadState()
            self._ExecuteCallback(callback, response)
            continue
        self.__progress = self.__GetLastByte(self._GetRangeHeaderFromResponse(response))
        if self.progress + 1 != self.stream.tell():
            raise exceptions.CommunicationError('Failed to transfer all bytes in chunk, upload paused at byte %d' % self.progress)
        self._ExecuteCallback(callback, response)
    if self.__complete and hasattr(self.stream, 'seek'):
        current_pos = self.stream.tell()
        self.stream.seek(0, os.SEEK_END)
        end_pos = self.stream.tell()
        self.stream.seek(current_pos)
        if current_pos != end_pos:
            raise exceptions.TransferInvalidError('Upload complete with %s additional bytes left in stream' % (int(end_pos) - int(current_pos)))
    self._ExecuteCallback(finish_callback, response)
    return response