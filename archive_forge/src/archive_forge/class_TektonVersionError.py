from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class TektonVersionError(exceptions.Error):
    """The Tekton version user supplied is not supported."""

    def __init__(self):
        msg = 'Tekton version is not supported. Only tekton.dev/v1beta1 is supported at the moment.'
        super(TektonVersionError, self).__init__(msg)