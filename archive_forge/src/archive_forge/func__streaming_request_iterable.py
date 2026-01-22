from __future__ import absolute_import
import google.api_core.gapic_v1.method
def _streaming_request_iterable(self, config, requests):
    """A generator that yields the config followed by the requests.

        Args:
            config (~.speech_v1.types.StreamingRecognitionConfig): The
                configuration to use for the stream.
            requests (Iterable[~.speech_v1.types.StreamingRecognizeRequest]):
                The input objects.

        Returns:
            Iterable[~.speech_v1.types.StreamingRecognizeRequest]): The
                correctly formatted input for
                :meth:`~.speech_v1.SpeechClient.streaming_recognize`.
        """
    yield {'streaming_config': config}
    for request in requests:
        yield request