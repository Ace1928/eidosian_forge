from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
Message sent by the client for the ``DeleteCustomClass`` method.

    Attributes:
        name (str):
            Required. The name of the custom class to delete. Format:

            ``projects/{project}/locations/{location}/customClasses/{custom_class}``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
    