from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from six.moves import urllib
def _MatchEncoding(self, recognizer, encoding, update=False, update_mask=None):
    """Matches encoding type based on auto or explicit decoding option."""
    if encoding is not None:
        if encoding == 'AUTO':
            recognizer.defaultRecognitionConfig.autoDecodingConfig = self._messages.AutoDetectDecodingConfig()
        elif encoding in EXPLICIT_ENCODING_OPTIONS:
            recognizer.defaultRecognitionConfig.explicitDecodingConfig = self._messages.ExplicitDecodingConfig()
            recognizer.defaultRecognitionConfig.explicitDecodingConfig.encoding = self._encoding_to_message[encoding]
        if update:
            if encoding == 'AUTO':
                update_mask.append('default_recognition_config.auto_decoding_config')
            else:
                update_mask.append('default_recognition_config.explicit_decoding_config')
    return (recognizer, update_mask)